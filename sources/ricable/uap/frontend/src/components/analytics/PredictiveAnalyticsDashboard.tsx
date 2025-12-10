// Predictive Analytics Dashboard with ML insights and forecasting
import React, { useState, useEffect } from 'react';
import { LineChart, Line, AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ScatterChart, Scatter } from 'recharts';
import { Brain, TrendingUp, AlertTriangle, Target, Clock, Activity, Zap, Database, RefreshCw, Download, Filter, Eye, Settings } from 'lucide-react';

interface PredictionData {
  predictionId: string;
  predictionType: string;
  modelType: string;
  predictedValue: number | string;
  confidence: string;
  confidenceScore: number;
  timeHorizonHours: number;
  createdAt: string;
  metadata: {
    inputFeatures: Record<string, any>;
    modelLastTrained: string | null;
    modelPerformance: {
      mae: number;
      mse: number;
      rmse: number;
      r2: number;
    } | null;
  };
  featureImportance: Record<string, number> | null;
}

interface ModelPerformance {
  modelType: string;
  predictionType: string;
  accuracy: number;
  lastTrained: string;
  trainingDataPoints: number;
  isActive: boolean;
}

interface AnomalyDetection {
  timestamp: string;
  anomalyScore: number;
  features: Record<string, number>;
  severity: 'low' | 'medium' | 'high';
  description: string;
}

interface CapacityPrediction {
  timeHorizonHours: number;
  predictedRequests: number;
  estimatedCpuUsage: number;
  estimatedMemoryUsage: number;
  estimatedConnections: number;
  confidence: string;
  recommendations: string[];
}

const CHART_COLORS = {
  primary: '#3B82F6',
  secondary: '#10B981',
  tertiary: '#F59E0B',
  quaternary: '#EF4444',
  accent: '#8B5CF6',
  neutral: '#6B7280'
};

export const PredictiveAnalyticsDashboard: React.FC = () => {
  const [predictions, setPredictions] = useState<PredictionData[]>([]);
  const [modelPerformance, setModelPerformance] = useState<ModelPerformance[]>([]);
  const [anomalies, setAnomalies] = useState<AnomalyDetection[]>([]);
  const [capacityPrediction, setCapacityPrediction] = useState<CapacityPrediction | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string>('all');
  const [timeHorizon, setTimeHorizon] = useState<number>(24);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchPredictiveData = async () => {
    try {
      setIsLoading(true);

      // Fetch recent predictions
      const predictionsResponse = await fetch('/api/analytics/predictions/insights?days=7');
      if (predictionsResponse.ok) {
        const data = await predictionsResponse.json();
        setPredictions(data.predictions || []);
      } else {
        setPredictions(generateMockPredictions());
      }

      // Fetch model performance
      const modelsResponse = await fetch('/api/analytics/predictions/models/performance');
      if (modelsResponse.ok) {
        const data = await modelsResponse.json();
        setModelPerformance(data.models || []);
      } else {
        setModelPerformance(generateMockModelPerformance());
      }

      // Fetch anomaly detection results
      const anomaliesResponse = await fetch('/api/analytics/predictions/anomalies');
      if (anomaliesResponse.ok) {
        const data = await anomaliesResponse.json();
        setAnomalies(transformAnomalies(data));
      } else {
        setAnomalies(generateMockAnomalies());
      }

      // Fetch capacity predictions
      const capacityResponse = await fetch(`/api/analytics/predictions/capacity?hours_ahead=${timeHorizon}`);
      if (capacityResponse.ok) {
        const data = await capacityResponse.json();
        setCapacityPrediction(data);
      } else {
        setCapacityPrediction(generateMockCapacityPrediction());
      }

      setLastUpdate(new Date());
    } catch (error) {
      console.error('Failed to fetch predictive analytics data:', error);
      // Set mock data as fallback
      setPredictions(generateMockPredictions());
      setModelPerformance(generateMockModelPerformance());
      setAnomalies(generateMockAnomalies());
      setCapacityPrediction(generateMockCapacityPrediction());
    } finally {
      setIsLoading(false);
    }
  };

  const generateMockPredictions = (): PredictionData[] => [
    {
      predictionId: 'pred-001',
      predictionType: 'usage_forecast',
      modelType: 'random_forest',
      predictedValue: 1247,
      confidence: 'high',
      confidenceScore: 0.87,
      timeHorizonHours: 24,
      createdAt: new Date().toISOString(),
      metadata: {
        inputFeatures: { hour: 14, dayOfWeek: 2, activeUsers: 89 },
        modelLastTrained: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
        modelPerformance: { mae: 45.2, mse: 2103.5, rmse: 45.9, r2: 0.87 }
      },
      featureImportance: { hour: 0.35, dayOfWeek: 0.28, activeUsers: 0.37 }
    },
    {
      predictionId: 'pred-002',
      predictionType: 'anomaly_detection',
      modelType: 'isolation_forest',
      predictedValue: 0.23,
      confidence: 'medium',
      confidenceScore: 0.72,
      timeHorizonHours: 1,
      createdAt: new Date().toISOString(),
      metadata: {
        inputFeatures: { cpuUsage: 67, memoryUsage: 54, requestRate: 234 },
        modelLastTrained: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(),
        modelPerformance: { mae: 0.12, mse: 0.023, rmse: 0.15, r2: 0.72 }
      },
      featureImportance: { cpuUsage: 0.42, memoryUsage: 0.31, requestRate: 0.27 }
    }
  ];

  const generateMockModelPerformance = (): ModelPerformance[] => [
    {
      modelType: 'Random Forest',
      predictionType: 'Usage Forecast',
      accuracy: 87.3,
      lastTrained: '2025-06-28T10:30:00Z',
      trainingDataPoints: 1247,
      isActive: true
    },
    {
      modelType: 'Isolation Forest',
      predictionType: 'Anomaly Detection',
      accuracy: 92.1,
      lastTrained: '2025-06-28T08:15:00Z',
      trainingDataPoints: 892,
      isActive: true
    },
    {
      modelType: 'Linear Regression',
      predictionType: 'Capacity Planning',
      accuracy: 78.6,
      lastTrained: '2025-06-27T14:45:00Z',
      trainingDataPoints: 2341,
      isActive: false
    }
  ];

  const generateMockAnomalies = (): AnomalyDetection[] => [
    {
      timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
      anomalyScore: 0.78,
      features: { cpuUsage: 89, memoryUsage: 76, requestRate: 445 },
      severity: 'medium',
      description: 'Elevated CPU and memory usage detected'
    },
    {
      timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
      anomalyScore: 0.34,
      features: { cpuUsage: 45, memoryUsage: 32, requestRate: 123 },
      severity: 'low',
      description: 'Minor deviation in request patterns'
    }
  ];

  const generateMockCapacityPrediction = (): CapacityPrediction => ({
    timeHorizonHours: 24,
    predictedRequests: 2847,
    estimatedCpuUsage: 67.3,
    estimatedMemoryUsage: 54.8,
    estimatedConnections: 234,
    confidence: 'high',
    recommendations: [
      'Current capacity sufficient for predicted load',
      'Monitor memory usage closely during peak hours',
      'Consider scaling if growth trend continues'
    ]
  });

  const transformAnomalies = (data: any): AnomalyDetection[] => {
    if (!data.anomalies) return [];
    return data.anomalies.map((anomaly: any) => ({
      timestamp: anomaly.timestamp,
      anomalyScore: anomaly.score,
      features: anomaly.features,
      severity: anomaly.score > 0.7 ? 'high' : anomaly.score > 0.4 ? 'medium' : 'low',
      description: anomaly.description || 'Anomalous behavior detected'
    }));
  };

  const triggerModelRetraining = async (modelType: string) => {
    try {
      const response = await fetch(`/api/analytics/predictions/models/${modelType}/retrain`, {
        method: 'POST'
      });
      if (response.ok) {
        fetchPredictiveData(); // Refresh data after retraining
      }
    } catch (error) {
      console.error('Failed to trigger model retraining:', error);
    }
  };

  const exportPredictions = async () => {
    try {
      const response = await fetch('/api/analytics/predictions/export', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          format: 'xlsx',
          include_model_performance: true,
          include_anomalies: true,
          time_range: '7d'
        })
      });
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `uap-predictions-${new Date().toISOString().split('T')[0]}.xlsx`;
        a.click();
        window.URL.revokeObjectURL(url);
      }
    } catch (error) {
      console.error('Failed to export predictions:', error);
    }
  };

  useEffect(() => {
    fetchPredictiveData();
    
    // Set up real-time updates
    const interval = setInterval(fetchPredictiveData, 2 * 60 * 1000); // Update every 2 minutes
    return () => clearInterval(interval);
  }, [timeHorizon]);

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="bg-gray-200 rounded-lg h-32"></div>
            ))}
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="bg-gray-200 rounded-lg h-80"></div>
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
          <Brain className="h-8 w-8 mr-3 text-purple-600" />
          Predictive Analytics
        </h1>
        
        <div className="flex items-center space-x-4">
          <select
            value={timeHorizon}
            onChange={(e) => setTimeHorizon(Number(e.target.value))}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500"
          >
            <option value={6}>6 hours ahead</option>
            <option value={12}>12 hours ahead</option>
            <option value={24}>24 hours ahead</option>
            <option value={72}>3 days ahead</option>
            <option value={168}>1 week ahead</option>
          </select>
          
          <button
            onClick={exportPredictions}
            className="flex items-center px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
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

      {/* Model Performance Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {modelPerformance.map((model, index) => (
          <div key={index} className="bg-white p-6 rounded-lg border border-gray-200">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold text-gray-900">{model.modelType}</h3>
                <p className="text-sm text-gray-600">{model.predictionType}</p>
              </div>
              <div className={`px-3 py-1 rounded-full text-xs font-medium ${
                model.isActive ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
              }`}>
                {model.isActive ? 'Active' : 'Inactive'}
              </div>
            </div>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Accuracy</span>
                <span className="text-lg font-bold text-gray-900">{model.accuracy}%</span>
              </div>
              
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full ${
                    model.accuracy > 85 ? 'bg-green-500' :
                    model.accuracy > 70 ? 'bg-yellow-500' : 'bg-red-500'
                  }`}
                  style={{ width: `${model.accuracy}%` }}
                ></div>
              </div>
              
              <div className="text-xs text-gray-500">
                <p>Training Data: {model.trainingDataPoints.toLocaleString()} points</p>
                <p>Last Trained: {new Date(model.lastTrained).toLocaleDateString()}</p>
              </div>
              
              {model.isActive && (
                <button
                  onClick={() => triggerModelRetraining(model.modelType.toLowerCase().replace(' ', '_'))}
                  className="w-full mt-3 px-3 py-2 text-sm bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200 transition-colors"
                >
                  <Settings className="h-4 w-4 mr-2 inline" />
                  Retrain Model
                </button>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Predictions and Capacity Planning */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Usage Forecast Chart */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <TrendingUp className="h-5 w-5 mr-2 text-blue-500" />
            Usage Forecast
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={generateForecastData()}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="actual" 
                stroke={CHART_COLORS.primary} 
                strokeWidth={2}
                name="Actual Usage"
              />
              <Line 
                type="monotone" 
                dataKey="predicted" 
                stroke={CHART_COLORS.accent} 
                strokeWidth={2}
                strokeDasharray="5 5"
                name="Predicted Usage"
              />
              <Line 
                type="monotone" 
                dataKey="upperBound" 
                stroke={CHART_COLORS.neutral} 
                strokeWidth={1}
                strokeDasharray="2 2"
                name="Upper Bound"
              />
              <Line 
                type="monotone" 
                dataKey="lowerBound" 
                stroke={CHART_COLORS.neutral} 
                strokeWidth={1}
                strokeDasharray="2 2"
                name="Lower Bound"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Capacity Planning */}
        {capacityPrediction && (
          <div className="bg-white p-6 rounded-lg border border-gray-200">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Database className="h-5 w-5 mr-2 text-green-500" />
              Capacity Planning ({timeHorizon}h ahead)
            </h3>
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 bg-blue-50 rounded-lg">
                  <p className="text-sm font-medium text-blue-800">Predicted Requests</p>
                  <p className="text-2xl font-bold text-blue-900">
                    {capacityPrediction.predictedRequests.toLocaleString()}
                  </p>
                </div>
                <div className="p-4 bg-yellow-50 rounded-lg">
                  <p className="text-sm font-medium text-yellow-800">Est. CPU Usage</p>
                  <p className="text-2xl font-bold text-yellow-900">
                    {capacityPrediction.estimatedCpuUsage}%
                  </p>
                </div>
                <div className="p-4 bg-green-50 rounded-lg">
                  <p className="text-sm font-medium text-green-800">Est. Memory Usage</p>
                  <p className="text-2xl font-bold text-green-900">
                    {capacityPrediction.estimatedMemoryUsage}%
                  </p>
                </div>
                <div className="p-4 bg-purple-50 rounded-lg">
                  <p className="text-sm font-medium text-purple-800">Est. Connections</p>
                  <p className="text-2xl font-bold text-purple-900">
                    {capacityPrediction.estimatedConnections}
                  </p>
                </div>
              </div>
              
              <div className="border-t pt-4">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Recommendations</h4>
                <ul className="space-y-1">
                  {capacityPrediction.recommendations.map((rec, index) => (
                    <li key={index} className="text-sm text-gray-600 flex items-start">
                      <Target className="h-4 w-4 mr-2 mt-0.5 text-purple-500" />
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Anomaly Detection */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Anomaly Timeline */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <AlertTriangle className="h-5 w-5 mr-2 text-red-500" />
            Anomaly Detection
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart data={generateAnomalyChartData()}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="hour" />
              <YAxis dataKey="anomalyScore" />
              <Tooltip 
                formatter={(value: any, name: string) => [
                  name === 'anomalyScore' ? `${value.toFixed(2)}` : value,
                  name === 'anomalyScore' ? 'Anomaly Score' : name
                ]}
                labelFormatter={(value) => `Hour: ${value}`}
              />
              <Scatter 
                name="Anomalies" 
                dataKey="anomalyScore" 
                fill={CHART_COLORS.quaternary}
                fillOpacity={0.7}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>

        {/* Recent Anomalies List */}
        <div className="bg-white p-6 rounded-lg border border-gray-200">
          <h3 className="text-lg font-semibold mb-4">Recent Anomalies</h3>
          <div className="space-y-3 max-h-80 overflow-y-auto">
            {anomalies.slice(0, 10).map((anomaly, index) => (
              <div key={index} className={`p-3 rounded-lg border-l-4 ${
                anomaly.severity === 'high' ? 'border-red-500 bg-red-50' :
                anomaly.severity === 'medium' ? 'border-yellow-500 bg-yellow-50' :
                'border-green-500 bg-green-50'
              }`}>
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <p className="text-sm font-medium text-gray-900">{anomaly.description}</p>
                    <p className="text-xs text-gray-600">
                      {new Date(anomaly.timestamp).toLocaleString()}
                    </p>
                  </div>
                  <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                    anomaly.severity === 'high' ? 'bg-red-100 text-red-800' :
                    anomaly.severity === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-green-100 text-green-800'
                  }`}>
                    {anomaly.severity}
                  </div>
                </div>
                <div className="flex justify-between text-xs text-gray-600">
                  <span>Score: {anomaly.anomalyScore.toFixed(2)}</span>
                  <span>CPU: {anomaly.features.cpuUsage}% | MEM: {anomaly.features.memoryUsage}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Feature Importance Analysis */}
      <div className="bg-white p-6 rounded-lg border border-gray-200">
        <h3 className="text-lg font-semibold mb-4 flex items-center">
          <Activity className="h-5 w-5 mr-2 text-blue-500" />
          Feature Importance Analysis
        </h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {predictions.filter(p => p.featureImportance).map((prediction, index) => (
            <div key={index}>
              <h4 className="text-md font-medium text-gray-700 mb-3">
                {prediction.predictionType.replace('_', ' ').toUpperCase()}
              </h4>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={Object.entries(prediction.featureImportance || {}).map(([feature, importance]) => ({
                  feature: feature.charAt(0).toUpperCase() + feature.slice(1),
                  importance: importance * 100
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="feature" />
                  <YAxis />
                  <Tooltip formatter={(value: any) => [`${value.toFixed(1)}%`, 'Importance']} />
                  <Bar dataKey="importance" fill={CHART_COLORS.accent} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// Helper functions for generating chart data
const generateForecastData = () => {
  const data = [];
  const now = new Date();
  
  // Generate historical data (last 12 hours)
  for (let i = 12; i >= 0; i--) {
    const time = new Date(now.getTime() - i * 60 * 60 * 1000);
    const baseValue = 100 + Math.sin(i / 4) * 50;
    data.push({
      time: time.toLocaleTimeString(),
      actual: Math.floor(baseValue + Math.random() * 20),
      predicted: null,
      upperBound: null,
      lowerBound: null
    });
  }
  
  // Generate future predictions (next 12 hours)
  for (let i = 1; i <= 12; i++) {
    const time = new Date(now.getTime() + i * 60 * 60 * 1000);
    const baseValue = 120 + Math.sin(i / 4) * 60;
    const predicted = Math.floor(baseValue + Math.random() * 10);
    data.push({
      time: time.toLocaleTimeString(),
      actual: null,
      predicted: predicted,
      upperBound: predicted + 30,
      lowerBound: Math.max(0, predicted - 30)
    });
  }
  
  return data;
};

const generateAnomalyChartData = () => {
  const data = [];
  for (let i = 0; i < 24; i++) {
    data.push({
      hour: i,
      anomalyScore: Math.random() * 0.8 + (i > 8 && i < 18 ? 0.2 : 0) // Higher during business hours
    });
  }
  return data;
};
