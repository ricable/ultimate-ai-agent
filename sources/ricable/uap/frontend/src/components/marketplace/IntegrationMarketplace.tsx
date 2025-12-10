// File: frontend/src/components/marketplace/IntegrationMarketplace.tsx
/**
 * Integration Marketplace component for discovering and managing integrations
 */

import React, { useState, useEffect } from 'react';
import { Search, Grid3x3, List, Filter, Star, ExternalLink, Settings, Download, Check } from 'lucide-react';

interface Integration {
  name: string;
  display_name: string;
  description: string;
  category: string;
  integration_type: string;
  auth_method: string;
  logo_url?: string;
  documentation_url?: string;
  pricing_model?: string;
  popularity_score: number;
  required_credentials: string[];
  supported_features: string[];
  webhook_events: string[];
  rate_limits: Record<string, number>;
}

interface MarketplaceStats {
  total_integrations: number;
  installed_integrations: number;
  active_integrations: number;
  categories: number;
  popular_integrations: string[];
}

interface Category {
  name: string;
  display_name: string;
  count: number;
  description?: string;
}

interface IntegrationStatus {
  integration_id: string;
  name: string;
  display_name: string;
  status: string;
  is_authenticated: boolean;
  created_at: string;
  last_error?: string;
}

export const IntegrationMarketplace: React.FC = () => {
  const [integrations, setIntegrations] = useState<Integration[]>([]);
  const [installedIntegrations, setInstalledIntegrations] = useState<IntegrationStatus[]>([]);
  const [categories, setCategories] = useState<Category[]>([]);
  const [stats, setStats] = useState<MarketplaceStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Filters and view state
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [showFilters, setShowFilters] = useState(false);
  const [sortBy, setSortBy] = useState<'popularity' | 'name' | 'category'>('popularity');
  
  // Modal state
  const [selectedIntegration, setSelectedIntegration] = useState<Integration | null>(null);
  const [showInstallModal, setShowInstallModal] = useState(false);
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [installCredentials, setInstallCredentials] = useState<Record<string, string>>({});

  useEffect(() => {
    loadMarketplaceData();
  }, []);

  useEffect(() => {
    loadIntegrations();
  }, [searchQuery, selectedCategory, sortBy]);

  const loadMarketplaceData = async () => {
    try {
      const [overviewResponse, categoriesResponse, installedResponse] = await Promise.all([
        fetch('/api/marketplace/', {
          headers: { 'Authorization': `Bearer ${localStorage.getItem('access_token')}` }
        }),
        fetch('/api/marketplace/categories', {
          headers: { 'Authorization': `Bearer ${localStorage.getItem('access_token')}` }
        }),
        fetch('/api/marketplace/installed', {
          headers: { 'Authorization': `Bearer ${localStorage.getItem('access_token')}` }
        })
      ]);

      if (overviewResponse.ok && categoriesResponse.ok && installedResponse.ok) {
        const overview = await overviewResponse.json();
        const categoriesData = await categoriesResponse.json();
        const installedData = await installedResponse.json();

        setStats(overview.stats);
        setCategories(categoriesData);
        setInstalledIntegrations(installedData);
      } else {
        throw new Error('Failed to load marketplace data');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    }
  };

  const loadIntegrations = async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      
      if (selectedCategory) params.append('category', selectedCategory);
      if (searchQuery) params.append('search', searchQuery);
      params.append('sort', sortBy);
      params.append('limit', '50');

      const response = await fetch(`/api/marketplace/integrations?${params}`, {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('access_token')}` }
      });

      if (response.ok) {
        const data = await response.json();
        setIntegrations(data);
      } else {
        throw new Error('Failed to load integrations');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load integrations');
    } finally {
      setLoading(false);
    }
  };

  const handleInstallIntegration = async (integration: Integration) => {
    try {
      const installData = {
        template_name: integration.name,
        config_overrides: {},
        custom_name: integration.display_name
      };

      const response = await fetch('/api/marketplace/install', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(installData)
      });

      if (response.ok) {
        const result = await response.json();
        setShowInstallModal(false);
        
        // If authentication is required, show config modal
        if (result.requires_authentication) {
          setSelectedIntegration(integration);
          setShowConfigModal(true);
        }
        
        // Reload data
        loadMarketplaceData();
      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Installation failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Installation failed');
    }
  };

  const handleConfigureIntegration = async (integrationId: string) => {
    try {
      const configData = {
        integration_id: integrationId,
        credentials: installCredentials,
        config_updates: {}
      };

      const response = await fetch('/api/marketplace/configure', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(configData)
      });

      if (response.ok) {
        setShowConfigModal(false);
        setInstallCredentials({});
        loadMarketplaceData();
      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Configuration failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Configuration failed');
    }
  };

  const isIntegrationInstalled = (integrationName: string) => {
    return installedIntegrations.some(installed => installed.name === integrationName);
  };

  const getIntegrationStatus = (integrationName: string) => {
    return installedIntegrations.find(installed => installed.name === integrationName);
  };

  const renderIntegrationCard = (integration: Integration) => {
    const installed = isIntegrationInstalled(integration.name);
    const status = getIntegrationStatus(integration.name);

    return (
      <div
        key={integration.name}
        className="bg-white rounded-lg border border-gray-200 hover:border-gray-300 hover:shadow-md transition-all duration-200 p-6"
      >
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center space-x-3">
            {integration.logo_url ? (
              <img
                src={integration.logo_url}
                alt={integration.display_name}
                className="w-10 h-10 rounded-lg object-cover"
              />
            ) : (
              <div className="w-10 h-10 rounded-lg bg-gray-100 flex items-center justify-center">
                <Settings className="w-5 h-5 text-gray-500" />
              </div>
            )}
            <div>
              <h3 className="font-semibold text-gray-900">{integration.display_name}</h3>
              <div className="flex items-center space-x-2 text-sm text-gray-500">
                <span className="capitalize">{integration.category.replace('_', ' ')}</span>
                <span>•</span>
                <div className="flex items-center">
                  <Star className="w-3 h-3 text-yellow-400 mr-1" />
                  <span>{integration.popularity_score}</span>
                </div>
              </div>
            </div>
          </div>
          
          {installed ? (
            <div className="flex items-center space-x-2">
              <Check className="w-4 h-4 text-green-500" />
              <span className="text-sm text-green-600 font-medium">
                {status?.is_authenticated ? 'Active' : 'Installed'}
              </span>
            </div>
          ) : (
            <button
              onClick={() => {
                setSelectedIntegration(integration);
                setShowInstallModal(true);
              }}
              className="flex items-center space-x-1 bg-blue-600 text-white px-3 py-1.5 rounded-md text-sm hover:bg-blue-700 transition-colors"
            >
              <Download className="w-3 h-3" />
              <span>Install</span>
            </button>
          )}
        </div>

        {/* Description */}
        <p className="text-gray-600 text-sm mb-4 line-clamp-2">{integration.description}</p>

        {/* Features */}
        <div className="mb-4">
          <div className="flex flex-wrap gap-1">
            {integration.supported_features.slice(0, 3).map((feature, index) => (
              <span
                key={index}
                className="inline-block bg-gray-100 text-gray-700 text-xs px-2 py-1 rounded"
              >
                {feature}
              </span>
            ))}
            {integration.supported_features.length > 3 && (
              <span className="text-gray-500 text-xs px-2 py-1">
                +{integration.supported_features.length - 3} more
              </span>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between pt-4 border-t border-gray-100">
          <div className="flex items-center space-x-2 text-sm text-gray-500">
            <span className="capitalize">{integration.auth_method.replace('_', ' ')}</span>
            {integration.pricing_model && (
              <>
                <span>•</span>
                <span>{integration.pricing_model}</span>
              </>
            )}
          </div>
          
          {integration.documentation_url && (
            <a
              href={integration.documentation_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center space-x-1 text-blue-600 hover:text-blue-800 text-sm"
            >
              <ExternalLink className="w-3 h-3" />
              <span>Docs</span>
            </a>
          )}
        </div>
      </div>
    );
  };

  const renderIntegrationList = (integration: Integration) => {
    const installed = isIntegrationInstalled(integration.name);
    const status = getIntegrationStatus(integration.name);

    return (
      <div
        key={integration.name}
        className="bg-white border border-gray-200 rounded-lg p-4 hover:border-gray-300 hover:shadow-sm transition-all duration-200"
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4 flex-1">
            {integration.logo_url ? (
              <img
                src={integration.logo_url}
                alt={integration.display_name}
                className="w-8 h-8 rounded-lg object-cover"
              />
            ) : (
              <div className="w-8 h-8 rounded-lg bg-gray-100 flex items-center justify-center">
                <Settings className="w-4 h-4 text-gray-500" />
              </div>
            )}
            
            <div className="flex-1">
              <div className="flex items-center space-x-2">
                <h3 className="font-medium text-gray-900">{integration.display_name}</h3>
                <span className="text-sm text-gray-500 capitalize">
                  {integration.category.replace('_', ' ')}
                </span>
                <div className="flex items-center">
                  <Star className="w-3 h-3 text-yellow-400 mr-1" />
                  <span className="text-sm text-gray-600">{integration.popularity_score}</span>
                </div>
              </div>
              <p className="text-sm text-gray-600 mt-1 line-clamp-1">{integration.description}</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-3">
            {integration.documentation_url && (
              <a
                href={integration.documentation_url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-blue-600"
              >
                <ExternalLink className="w-4 h-4" />
              </a>
            )}
            
            {installed ? (
              <div className="flex items-center space-x-2">
                <Check className="w-4 h-4 text-green-500" />
                <span className="text-sm text-green-600 font-medium">
                  {status?.is_authenticated ? 'Active' : 'Installed'}
                </span>
              </div>
            ) : (
              <button
                onClick={() => {
                  setSelectedIntegration(integration);
                  setShowInstallModal(true);
                }}
                className="flex items-center space-x-1 bg-blue-600 text-white px-3 py-1.5 rounded-md text-sm hover:bg-blue-700 transition-colors"
              >
                <Download className="w-3 h-3" />
                <span>Install</span>
              </button>
            )}
          </div>
        </div>
      </div>
    );
  };

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex">
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">Error</h3>
              <div className="mt-2 text-sm text-red-700">{error}</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Integration Marketplace</h1>
        <p className="text-gray-600">
          Discover and install integrations to extend your UAP capabilities
        </p>
        
        {/* Stats */}
        {stats && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <div className="text-2xl font-bold text-gray-900">{stats.total_integrations}</div>
              <div className="text-sm text-gray-600">Available Integrations</div>
            </div>
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <div className="text-2xl font-bold text-green-600">{stats.installed_integrations}</div>
              <div className="text-sm text-gray-600">Installed</div>
            </div>
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <div className="text-2xl font-bold text-blue-600">{stats.active_integrations}</div>
              <div className="text-sm text-gray-600">Active</div>
            </div>
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <div className="text-2xl font-bold text-purple-600">{stats.categories}</div>
              <div className="text-sm text-gray-600">Categories</div>
            </div>
          </div>
        )}
      </div>

      {/* Search and Filters */}
      <div className="bg-white rounded-lg border border-gray-200 p-4 mb-6">
        <div className="flex flex-col sm:flex-row gap-4">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
            <input
              type="text"
              placeholder="Search integrations..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          
          {/* Category Filter */}
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="">All Categories</option>
            {categories.map((category) => (
              <option key={category.name} value={category.name}>
                {category.display_name} ({category.count})
              </option>
            ))}
          </select>
          
          {/* Sort */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as 'popularity' | 'name' | 'category')}
            className="px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="popularity">Sort by Popularity</option>
            <option value="name">Sort by Name</option>
            <option value="category">Sort by Category</option>
          </select>
          
          {/* View Mode */}
          <div className="flex border border-gray-300 rounded-md">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 ${viewMode === 'grid' ? 'bg-blue-600 text-white' : 'text-gray-600 hover:bg-gray-50'}`}
            >
              <Grid3x3 className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 ${viewMode === 'list' ? 'bg-blue-600 text-white' : 'text-gray-600 hover:bg-gray-50'}`}
            >
              <List className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Loading */}
      {loading && (
        <div className="flex justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      )}

      {/* Integrations Grid/List */}
      {!loading && (
        <div className={
          viewMode === 'grid' 
            ? "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
            : "space-y-4"
        }>
          {integrations.map((integration) => (
            viewMode === 'grid' 
              ? renderIntegrationCard(integration)
              : renderIntegrationList(integration)
          ))}
        </div>
      )}

      {/* Empty State */}
      {!loading && integrations.length === 0 && (
        <div className="text-center py-12">
          <Settings className="mx-auto h-12 w-12 text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No integrations found</h3>
          <p className="text-gray-600">
            {searchQuery || selectedCategory 
              ? "Try adjusting your search or filters"
              : "No integrations are available at this time"
            }
          </p>
        </div>
      )}

      {/* Install Modal */}
      {showInstallModal && selectedIntegration && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-md w-full p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              Install {selectedIntegration.display_name}
            </h3>
            
            <div className="space-y-4">
              <p className="text-gray-600">{selectedIntegration.description}</p>
              
              <div>
                <h4 className="font-medium text-gray-900 mb-2">Features:</h4>
                <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                  {selectedIntegration.supported_features.map((feature, index) => (
                    <li key={index}>{feature}</li>
                  ))}
                </ul>
              </div>
              
              <div>
                <h4 className="font-medium text-gray-900 mb-2">Required Credentials:</h4>
                <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                  {selectedIntegration.required_credentials.map((cred, index) => (
                    <li key={index}>{cred}</li>
                  ))}
                </ul>
              </div>
            </div>
            
            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => setShowInstallModal(false)}
                className="px-4 py-2 text-gray-700 border border-gray-300 rounded-md hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={() => handleInstallIntegration(selectedIntegration)}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
              >
                Install
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Configuration Modal */}
      {showConfigModal && selectedIntegration && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-md w-full p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              Configure {selectedIntegration.display_name}
            </h3>
            
            <div className="space-y-4">
              <p className="text-gray-600">
                Enter your credentials to authenticate with {selectedIntegration.display_name}.
              </p>
              
              {selectedIntegration.required_credentials.map((credential, index) => (
                <div key={index}>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {credential.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </label>
                  <input
                    type={credential.toLowerCase().includes('secret') || credential.toLowerCase().includes('token') ? 'password' : 'text'}
                    value={installCredentials[credential] || ''}
                    onChange={(e) => setInstallCredentials(prev => ({
                      ...prev,
                      [credential]: e.target.value
                    }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder={`Enter ${credential}`}
                  />
                </div>
              ))}
            </div>
            
            <div className="flex justify-end space-x-3 mt-6">
              <button
                onClick={() => {
                  setShowConfigModal(false);
                  setInstallCredentials({});
                }}
                className="px-4 py-2 text-gray-700 border border-gray-300 rounded-md hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  const installed = installedIntegrations.find(i => i.name === selectedIntegration.name);
                  if (installed) {
                    handleConfigureIntegration(installed.integration_id);
                  }
                }}
                disabled={!selectedIntegration.required_credentials.every(cred => installCredentials[cred])}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Configure
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};