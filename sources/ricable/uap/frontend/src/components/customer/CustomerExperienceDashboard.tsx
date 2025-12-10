// Customer Experience Dashboard - Main dashboard for customer experience management
import React, { useState, useEffect } from 'react';
import { 
  Users, Heart, TrendingUp, AlertTriangle, Star, MessageCircle, 
  Target, Activity, Clock, ChevronRight, Filter, Download, RefreshCw 
} from 'lucide-react';
import { Card } from '../ui/Card';

interface CustomerProfile {
  id: string;
  segment: string;
  tier: string;
  lifetime_value: number;
  current_stage: string;
  activity_score: number;
  churn_risk_score: number;
}

interface SatisfactionMetrics {
  csat_score: number;
  nps_score: number;
  satisfaction_trend: number[];
}

interface JourneyMetrics {
  current_stage: string;
  days_in_stage: number;
  conversion_rate: number;
  engagement_score: number;
}

interface CustomerDashboardData {
  customer_profile: CustomerProfile;
  satisfaction_metrics: SatisfactionMetrics;
  journey_metrics: JourneyMetrics;
  recent_interactions: Array<{
    id: string;
    type: string;
    sentiment: string;
    satisfaction: number;
    timestamp: string;
  }>;
  active_recommendations: Array<{
    id: string;
    type: string;
    title: string;
    confidence: number;
    created_at: string;
  }>;
}

interface AnalyticsOverview {
  overview: {
    total_customers: number;
    high_risk_customers: number;
    medium_risk_customers: number;
    avg_satisfaction: number;
    avg_nps: number;
  };
  satisfaction_benchmarks: {
    avg_csat: number;
    avg_nps: number;
    avg_ces: number;
    satisfaction_rate: number;
  };
  stage_distribution: Record<string, number>;
}

export const CustomerExperienceDashboard: React.FC = () => {
  const [selectedCustomer, setSelectedCustomer] = useState<string | null>(null);
  const [customerData, setCustomerData] = useState<CustomerDashboardData | null>(null);
  const [analyticsData, setAnalyticsData] = useState<AnalyticsOverview | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [timeRange, setTimeRange] = useState(30);
  const [view, setView] = useState<'overview' | 'customer' | 'analytics'>('overview');

  useEffect(() => {
    fetchAnalyticsData();
  }, [timeRange]);

  useEffect(() => {
    if (selectedCustomer) {
      fetchCustomerData(selectedCustomer);
    }
  }, [selectedCustomer]);

  const fetchAnalyticsData = async () => {
    try {
      const response = await fetch(`/api/customer-experience/analytics/overview?days=${timeRange}`);
      if (response.ok) {
        const data = await response.json();
        setAnalyticsData(data);
      } else {
        // Mock data for demo
        setAnalyticsData({
          overview: {
            total_customers: 1250,
            high_risk_customers: 45,
            medium_risk_customers: 180,
            avg_satisfaction: 4.2,
            avg_nps: 7.8
          },
          satisfaction_benchmarks: {
            avg_csat: 4.2,
            avg_nps: 7.8,
            avg_ces: 5.6,
            satisfaction_rate: 0.84
          },
          stage_distribution: {
            awareness: 120,
            consideration: 200,
            trial: 150,
            active_use: 600,
            renewal: 100,
            expansion: 80
          }
        });
      }
    } catch (error) {
      console.error('Failed to fetch analytics data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchCustomerData = async (customerId: string) => {
    try {
      const response = await fetch(`/api/customer-experience/dashboard/${customerId}`);
      if (response.ok) {
        const data = await response.json();
        setCustomerData(data);
      } else {
        // Mock data for demo
        setCustomerData({
          customer_profile: {
            id: customerId,
            segment: 'enterprise',
            tier: 'gold',
            lifetime_value: 25000,
            current_stage: 'active_use',
            activity_score: 85,
            churn_risk_score: 0.2
          },
          satisfaction_metrics: {
            csat_score: 4.5,
            nps_score: 9,
            satisfaction_trend: [4.0, 4.2, 4.3, 4.5, 4.4]
          },
          journey_metrics: {
            current_stage: 'active_use',
            days_in_stage: 45,
            conversion_rate: 0.85,
            engagement_score: 82
          },
          recent_interactions: [
            {
              id: '1',
              type: 'chat',
              sentiment: 'positive',
              satisfaction: 5,
              timestamp: '2024-01-15T10:30:00Z'
            }
          ],
          active_recommendations: [
            {
              id: '1',
              type: 'feature',
              title: 'Try Advanced Analytics',
              confidence: 0.85,
              created_at: '2024-01-14T08:00:00Z'
            }
          ]
        });
      }
    } catch (error) {
      console.error('Failed to fetch customer data:', error);
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive': return 'text-green-600';
      case 'negative': return 'text-red-600';
      case 'frustrated': return 'text-orange-600';
      default: return 'text-gray-600';
    }
  };

  const getRiskColor = (score: number) => {
    if (score > 0.7) return 'text-red-600';
    if (score > 0.4) return 'text-orange-600';
    return 'text-green-600';
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0
    }).format(amount);
  };

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="bg-gray-200 rounded-lg h-32"></div>
            ))}
          </div>
          <div className="bg-gray-200 rounded-lg h-96"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Customer Experience</h1>
          <p className="text-gray-500 mt-1">Intelligent customer support and engagement management</p>
        </div>
        
        <div className="flex items-center space-x-4">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(Number(e.target.value))}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value={7}>Last 7 days</option>
            <option value={30}>Last 30 days</option>
            <option value={90}>Last 90 days</option>
          </select>
          
          <button
            onClick={() => fetchAnalyticsData()}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Refresh
          </button>
        </div>
      </div>

      {/* Navigation */}
      <div className="flex space-x-4 border-b border-gray-200">
        {[
          { id: 'overview', label: 'Overview', icon: Activity },
          { id: 'customer', label: 'Customer Details', icon: Users },
          { id: 'analytics', label: 'Analytics', icon: TrendingUp }
        ].map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setView(id as any)}
            className={`flex items-center px-4 py-2 border-b-2 font-medium text-sm ${
              view === id
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            <Icon className="h-4 w-4 mr-2" />
            {label}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {view === 'overview' && analyticsData && (
        <div className="space-y-6">
          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <Card className="p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <Users className="h-8 w-8 text-blue-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Total Customers</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {analyticsData.overview.total_customers.toLocaleString()}
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <Heart className="h-8 w-8 text-green-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Avg Satisfaction</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {analyticsData.overview.avg_satisfaction.toFixed(1)}/5
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <Star className="h-8 w-8 text-yellow-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">Net Promoter Score</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {analyticsData.overview.avg_nps.toFixed(1)}
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-6">
              <div className="flex items-center">
                <div className="flex-shrink-0">
                  <AlertTriangle className="h-8 w-8 text-red-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-500">High Risk Customers</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {analyticsData.overview.high_risk_customers}
                  </p>
                </div>
              </div>
            </Card>
          </div>

          {/* Journey Stage Distribution */}
          <Card className="p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Customer Journey Distribution</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
              {Object.entries(analyticsData.stage_distribution).map(([stage, count]) => (
                <div key={stage} className="text-center">
                  <div className="text-2xl font-bold text-gray-900">{count}</div>
                  <div className="text-sm text-gray-500 capitalize">{stage.replace('_', ' ')}</div>
                </div>
              ))}
            </div>
          </Card>

          {/* Sample Customer List */}
          <Card className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900">Recent Customer Activity</h3>
              <button
                className="text-blue-600 hover:text-blue-700 text-sm font-medium"
                onClick={() => setView('customer')}
              >
                View All <ChevronRight className="h-4 w-4 inline ml-1" />
              </button>
            </div>
            
            <div className="space-y-3">
              {[
                { id: 'cust-1', name: 'Acme Corp', segment: 'Enterprise', risk: 0.2, ltv: 45000 },
                { id: 'cust-2', name: 'Tech Solutions Inc', segment: 'SMB', risk: 0.8, ltv: 12000 },
                { id: 'cust-3', name: 'Global Industries', segment: 'Enterprise', risk: 0.3, ltv: 78000 }
              ].map((customer) => (
                <div
                  key={customer.id}
                  className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50 cursor-pointer"
                  onClick={() => {
                    setSelectedCustomer(customer.id);
                    setView('customer');
                  }}
                >
                  <div className="flex items-center">
                    <div className="flex-shrink-0">
                      <div className="h-10 w-10 bg-blue-500 rounded-full flex items-center justify-center">
                        <span className="text-white font-medium">
                          {customer.name.substring(0, 2).toUpperCase()}
                        </span>
                      </div>
                    </div>
                    <div className="ml-4">
                      <p className="text-sm font-medium text-gray-900">{customer.name}</p>
                      <p className="text-sm text-gray-500">{customer.segment}</p>
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-4">
                    <div className="text-right">
                      <p className="text-sm font-medium text-gray-900">{formatCurrency(customer.ltv)}</p>
                      <p className="text-xs text-gray-500">Lifetime Value</p>
                    </div>
                    
                    <div className={`px-2 py-1 rounded-full text-xs font-medium ${
                      customer.risk > 0.7
                        ? 'bg-red-100 text-red-800'
                        : customer.risk > 0.4
                        ? 'bg-yellow-100 text-yellow-800'
                        : 'bg-green-100 text-green-800'
                    }`}>
                      {customer.risk > 0.7 ? 'High Risk' : customer.risk > 0.4 ? 'Medium Risk' : 'Low Risk'}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      )}

      {/* Customer Details Tab */}
      {view === 'customer' && (
        <div className="space-y-6">
          {!selectedCustomer ? (
            <Card className="p-8 text-center">
              <Users className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">Select a Customer</h3>
              <p className="text-gray-500">Choose a customer to view detailed experience data</p>
              
              <div className="mt-6">
                <input
                  type="text"
                  placeholder="Enter customer ID"
                  className="px-4 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  onKeyPress={(e) => {
                    if (e.key === 'Enter') {
                      setSelectedCustomer((e.target as HTMLInputElement).value);
                    }
                  }}
                />
                <button
                  onClick={() => {
                    const input = document.querySelector('input[placeholder="Enter customer ID"]') as HTMLInputElement;
                    if (input.value) {
                      setSelectedCustomer(input.value);
                    }
                  }}
                  className="ml-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                >
                  Load Customer
                </button>
              </div>
            </Card>
          ) : customerData ? (
            <div className="space-y-6">
              {/* Customer Header */}
              <Card className="p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div className="h-16 w-16 bg-blue-500 rounded-full flex items-center justify-center">
                      <span className="text-white text-xl font-bold">
                        {customerData.customer_profile.segment.substring(0, 2).toUpperCase()}
                      </span>
                    </div>
                    <div className="ml-4">
                      <h2 className="text-xl font-bold text-gray-900">Customer {selectedCustomer}</h2>
                      <p className="text-gray-500">
                        {customerData.customer_profile.segment} • {customerData.customer_profile.tier} tier
                      </p>
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <p className="text-2xl font-bold text-gray-900">
                      {formatCurrency(customerData.customer_profile.lifetime_value)}
                    </p>
                    <p className="text-sm text-gray-500">Lifetime Value</p>
                  </div>
                </div>
              </Card>

              {/* Key Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <Card className="p-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">
                      {customerData.satisfaction_metrics.csat_score.toFixed(1)}
                    </div>
                    <div className="text-sm text-gray-500">CSAT Score</div>
                  </div>
                </Card>

                <Card className="p-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {customerData.satisfaction_metrics.nps_score}
                    </div>
                    <div className="text-sm text-gray-500">NPS Score</div>
                  </div>
                </Card>

                <Card className="p-4">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-purple-600">
                      {customerData.customer_profile.activity_score}%
                    </div>
                    <div className="text-sm text-gray-500">Activity Score</div>
                  </div>
                </Card>

                <Card className="p-4">
                  <div className="text-center">
                    <div className={`text-2xl font-bold ${getRiskColor(customerData.customer_profile.churn_risk_score)}`}>
                      {(customerData.customer_profile.churn_risk_score * 100).toFixed(0)}%
                    </div>
                    <div className="text-sm text-gray-500">Churn Risk</div>
                  </div>
                </Card>
              </div>

              {/* Journey & Interactions */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card className="p-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Customer Journey</h3>
                  <div className="space-y-4">
                    <div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-500">Current Stage</span>
                        <span className="text-sm font-medium text-gray-900 capitalize">
                          {customerData.journey_metrics.current_stage.replace('_', ' ')}
                        </span>
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-500">Days in Stage</span>
                        <span className="text-sm font-medium text-gray-900">
                          {customerData.journey_metrics.days_in_stage} days
                        </span>
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-gray-500">Engagement Score</span>
                        <span className="text-sm font-medium text-gray-900">
                          {customerData.journey_metrics.engagement_score}%
                        </span>
                      </div>
                    </div>
                  </div>
                </Card>

                <Card className="p-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-4">Recent Interactions</h3>
                  <div className="space-y-3">
                    {customerData.recent_interactions.map((interaction) => (
                      <div key={interaction.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div className="flex items-center">
                          <MessageCircle className="h-4 w-4 text-gray-400 mr-2" />
                          <div>
                            <p className="text-sm font-medium text-gray-900 capitalize">{interaction.type}</p>
                            <p className={`text-xs ${getSentimentColor(interaction.sentiment)}`}>
                              {interaction.sentiment || 'neutral'}
                            </p>
                          </div>
                        </div>
                        
                        <div className="text-right">
                          <div className="flex items-center">
                            {[...Array(5)].map((_, i) => (
                              <Star
                                key={i}
                                className={`h-3 w-3 ${
                                  i < interaction.satisfaction ? 'text-yellow-400 fill-current' : 'text-gray-300'
                                }`}
                              />
                            ))}
                          </div>
                          <p className="text-xs text-gray-500">
                            {new Date(interaction.timestamp).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </Card>
              </div>

              {/* Recommendations */}
              <Card className="p-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Active Recommendations</h3>
                <div className="space-y-3">
                  {customerData.active_recommendations.map((rec) => (
                    <div key={rec.id} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                      <div className="flex items-center">
                        <Target className="h-5 w-5 text-blue-500 mr-3" />
                        <div>
                          <p className="font-medium text-gray-900">{rec.title}</p>
                          <p className="text-sm text-gray-500 capitalize">{rec.type} recommendation</p>
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <p className="text-sm font-medium text-gray-900">
                          {(rec.confidence * 100).toFixed(0)}% confidence
                        </p>
                        <p className="text-xs text-gray-500">
                          {new Date(rec.created_at).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </Card>
            </div>
          ) : (
            <Card className="p-8 text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-500">Loading customer data...</p>
            </Card>
          )}
        </div>
      )}

      {/* Analytics Tab */}
      {view === 'analytics' && analyticsData && (
        <div className="space-y-6">
          <Card className="p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Satisfaction Benchmarks</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">
                  {analyticsData.satisfaction_benchmarks.avg_csat.toFixed(1)}
                </div>
                <div className="text-sm text-gray-500">Average CSAT</div>
              </div>
              
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">
                  {analyticsData.satisfaction_benchmarks.avg_nps.toFixed(1)}
                </div>
                <div className="text-sm text-gray-500">Average NPS</div>
              </div>
              
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {analyticsData.satisfaction_benchmarks.avg_ces.toFixed(1)}
                </div>
                <div className="text-sm text-gray-500">Average CES</div>
              </div>
              
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">
                  {(analyticsData.satisfaction_benchmarks.satisfaction_rate * 100).toFixed(0)}%
                </div>
                <div className="text-sm text-gray-500">Satisfaction Rate</div>
              </div>
            </div>
          </Card>

          {/* Add more analytics visualizations here */}
          <Card className="p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Coming Soon</h3>
            <p className="text-gray-500">Advanced analytics charts and insights will be available here.</p>
          </Card>
        </div>
      )}
    </div>
  );
};