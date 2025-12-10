/**
 * Workflow Marketplace Component
 * 
 * Browse, search, and install workflow templates from the marketplace
 * with ratings, categories, and filtering capabilities.
 */

import React, { useState, useEffect } from 'react';
import {
  ArrowLeft,
  Search,
  Filter,
  Star,
  Download,
  Eye,
  Grid,
  List,
  Tag,
  Heart,
  Award,
  TrendingUp,
  Clock,
  DollarSign,
  Check
} from 'lucide-react';

interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  subcategory?: string;
  version: string;
  tags: string[];
  keywords: string[];
  documentation: string;
  created_by: string;
  organization_id: string;
  created_at: string;
  updated_at: string;
  is_featured: boolean;
  is_verified: boolean;
  price: number;
  download_count: number;
  rating: number;
  rating_count: number;
  definition: any;
  variables: any;
}

interface WorkflowMarketplaceProps {
  onBack: () => void;
  onInstall: (templateId: string, workflowName: string) => void;
}

type ViewMode = 'grid' | 'list';
type SortBy = 'relevance' | 'rating' | 'downloads' | 'date' | 'price';

interface Filters {
  query: string;
  category: string;
  subcategory: string;
  tags: string[];
  minRating: number;
  maxPrice: number;
  isFreeOnly: boolean;
  isFeatured: boolean;
  isVerified: boolean;
  sortBy: SortBy;
}

export const WorkflowMarketplace: React.FC<WorkflowMarketplaceProps> = ({
  onBack,
  onInstall
}) => {
  const [templates, setTemplates] = useState<WorkflowTemplate[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedTemplate, setSelectedTemplate] = useState<WorkflowTemplate | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('grid');
  const [showFilters, setShowFilters] = useState(false);
  const [categories, setCategories] = useState<Record<string, string[]>>({});
  const [stats, setStats] = useState<any>({});
  
  const [filters, setFilters] = useState<Filters>({
    query: '',
    category: '',
    subcategory: '',
    tags: [],
    minRating: 0,
    maxPrice: 0,
    isFreeOnly: false,
    isFeatured: false,
    isVerified: false,
    sortBy: 'relevance'
  });

  useEffect(() => {
    loadTemplates();
    loadCategories();
    loadStats();
  }, [filters]);

  const loadTemplates = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      
      if (filters.query) params.append('query', filters.query);
      if (filters.category) params.append('category', filters.category);
      if (filters.subcategory) params.append('subcategory', filters.subcategory);
      if (filters.tags.length > 0) params.append('tags', filters.tags.join(','));
      if (filters.minRating > 0) params.append('min_rating', filters.minRating.toString());
      if (filters.maxPrice > 0) params.append('max_price', filters.maxPrice.toString());
      if (filters.isFreeOnly) params.append('is_free_only', 'true');
      if (filters.isFeatured) params.append('is_featured', 'true');
      if (filters.isVerified) params.append('is_verified', 'true');
      params.append('sort_by', filters.sortBy);
      
      const response = await fetch(`/api/workflows/marketplace/templates?${params}`);
      const data = await response.json();
      
      setTemplates(data.templates || []);
    } catch (error) {
      console.error('Failed to load templates:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadCategories = async () => {
    try {
      const response = await fetch('/api/workflows/marketplace/categories');
      const data = await response.json();
      setCategories(data);
    } catch (error) {
      console.error('Failed to load categories:', error);
    }
  };

  const loadStats = async () => {
    try {
      const response = await fetch('/api/workflows/marketplace/stats');
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  const handleInstall = async (template: WorkflowTemplate) => {
    const workflowName = prompt('Enter a name for your new workflow:', `${template.name} (Copy)`);
    if (!workflowName) return;
    
    try {
      const response = await fetch(`/api/workflows/marketplace/templates/${template.id}/install`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ workflow_name: workflowName })
      });
      
      if (response.ok) {
        onInstall(template.id, workflowName);
      } else {
        throw new Error('Failed to install template');
      }
    } catch (error) {
      console.error('Failed to install template:', error);
      alert('Failed to install template. Please try again.');
    }
  };

  const handleRateTemplate = async (templateId: string, rating: number) => {
    try {
      await fetch(`/api/workflows/marketplace/templates/${templateId}/rate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rating })
      });
      
      loadTemplates(); // Refresh to show updated rating
    } catch (error) {
      console.error('Failed to rate template:', error);
    }
  };

  const updateFilters = (updates: Partial<Filters>) => {
    setFilters(prev => ({ ...prev, ...updates }));
  };

  const renderHeader = () => (
    <div className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={onBack}
            className="p-2 hover:bg-gray-100 rounded-lg"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Workflow Marketplace</h1>
            <p className="text-gray-600">Discover and install workflow templates</p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <div className="flex bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setViewMode('grid')}
              className={`px-3 py-1 rounded-md ${
                viewMode === 'grid'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <Grid className="h-4 w-4" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`px-3 py-1 rounded-md ${
                viewMode === 'list'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              <List className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>
      
      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mt-6">
        <div className="bg-blue-50 rounded-lg p-4">
          <div className="flex items-center">
            <div className="p-2 bg-blue-100 rounded-lg">
              <Award className="h-6 w-6 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-blue-600">Total Templates</p>
              <p className="text-2xl font-bold text-blue-900">{stats.total_templates || 0}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-green-50 rounded-lg p-4">
          <div className="flex items-center">
            <div className="p-2 bg-green-100 rounded-lg">
              <Download className="h-6 w-6 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-green-600">Total Downloads</p>
              <p className="text-2xl font-bold text-green-900">{stats.total_downloads || 0}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-yellow-50 rounded-lg p-4">
          <div className="flex items-center">
            <div className="p-2 bg-yellow-100 rounded-lg">
              <Star className="h-6 w-6 text-yellow-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-yellow-600">Average Rating</p>
              <p className="text-2xl font-bold text-yellow-900">{stats.average_rating || 0}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-purple-50 rounded-lg p-4">
          <div className="flex items-center">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Heart className="h-6 w-6 text-purple-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-purple-600">Featured</p>
              <p className="text-2xl font-bold text-purple-900">{stats.featured_templates || 0}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderSearchAndFilters = () => (
    <div className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
          <input
            type="text"
            placeholder="Search templates..."
            value={filters.query}
            onChange={(e) => updateFilters({ query: e.target.value })}
            className="pl-10 pr-4 py-2 w-full border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
        
        <button
          onClick={() => setShowFilters(!showFilters)}
          className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 flex items-center gap-2"
        >
          <Filter className="h-4 w-4" />
          Filters
        </button>
      </div>
      
      {showFilters && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Category</label>
              <select
                value={filters.category}
                onChange={(e) => updateFilters({ category: e.target.value, subcategory: '' })}
                className="w-full border border-gray-300 rounded-lg px-3 py-2"
              >
                <option value="">All Categories</option>
                {Object.keys(categories).map(category => (
                  <option key={category} value={category}>{category}</option>
                ))}
              </select>
            </div>
            
            {filters.category && categories[filters.category] && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Subcategory</label>
                <select
                  value={filters.subcategory}
                  onChange={(e) => updateFilters({ subcategory: e.target.value })}
                  className="w-full border border-gray-300 rounded-lg px-3 py-2"
                >
                  <option value="">All Subcategories</option>
                  {categories[filters.category].map(subcategory => (
                    <option key={subcategory} value={subcategory}>{subcategory}</option>
                  ))}
                </select>
              </div>
            )}
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Sort By</label>
              <select
                value={filters.sortBy}
                onChange={(e) => updateFilters({ sortBy: e.target.value as SortBy })}
                className="w-full border border-gray-300 rounded-lg px-3 py-2"
              >
                <option value="relevance">Relevance</option>
                <option value="rating">Rating</option>
                <option value="downloads">Downloads</option>
                <option value="date">Date</option>
                <option value="price">Price</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Filters</label>
              <div className="space-y-2">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={filters.isFreeOnly}
                    onChange={(e) => updateFilters({ isFreeOnly: e.target.checked })}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="ml-2 text-sm text-gray-700">Free only</span>
                </label>
                
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={filters.isFeatured}
                    onChange={(e) => updateFilters({ isFeatured: e.target.checked })}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="ml-2 text-sm text-gray-700">Featured</span>
                </label>
                
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={filters.isVerified}
                    onChange={(e) => updateFilters({ isVerified: e.target.checked })}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="ml-2 text-sm text-gray-700">Verified</span>
                </label>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const renderTemplateCard = (template: WorkflowTemplate) => (
    <div key={template.id} className="bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow overflow-hidden">
      <div className="p-6">
        <div className="flex items-start justify-between mb-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2">
              <h3 className="text-lg font-semibold text-gray-900 truncate">
                {template.name}
              </h3>
              {template.is_featured && (
                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
                  <Award className="h-3 w-3 mr-1" />
                  Featured
                </span>
              )}
              {template.is_verified && (
                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                  <Check className="h-3 w-3 mr-1" />
                  Verified
                </span>
              )}
            </div>
            
            <p className="text-sm text-gray-600 line-clamp-2 mb-3">
              {template.description}
            </p>
            
            <div className="flex items-center gap-4 text-sm text-gray-500 mb-3">
              <div className="flex items-center gap-1">
                <Star className="h-4 w-4 fill-current text-yellow-400" />
                <span>{template.rating.toFixed(1)}</span>
                <span>({template.rating_count})</span>
              </div>
              
              <div className="flex items-center gap-1">
                <Download className="h-4 w-4" />
                <span>{template.download_count}</span>
              </div>
              
              {template.price > 0 ? (
                <div className="flex items-center gap-1">
                  <DollarSign className="h-4 w-4" />
                  <span>${(template.price / 100).toFixed(2)}</span>
                </div>
              ) : (
                <span className="text-green-600 font-medium">Free</span>
              )}
            </div>
            
            <div className="flex items-center gap-2 mb-4">
              <span className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-blue-100 text-blue-800">
                {template.category}
              </span>
              {template.subcategory && (
                <span className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-gray-100 text-gray-800">
                  {template.subcategory}
                </span>
              )}
            </div>
            
            {template.tags && template.tags.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {template.tags.slice(0, 3).map((tag, index) => (
                  <span
                    key={index}
                    className="inline-flex items-center px-2 py-1 rounded-md text-xs bg-gray-100 text-gray-700"
                  >
                    <Tag className="h-3 w-3 mr-1" />
                    {tag}
                  </span>
                ))}
                {template.tags.length > 3 && (
                  <span className="text-xs text-gray-500">
                    +{template.tags.length - 3} more
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
      
      <div className="px-6 py-4 bg-gray-50 border-t border-gray-200">
        <div className="flex items-center justify-between">
          <button
            onClick={() => setSelectedTemplate(template)}
            className="px-3 py-1.5 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
          >
            <Eye className="h-3 w-3 mr-1 inline" />
            Preview
          </button>
          
          <button
            onClick={() => handleInstall(template)}
            className="px-4 py-1.5 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700"
          >
            <Download className="h-3 w-3 mr-1 inline" />
            Install
          </button>
        </div>
      </div>
    </div>
  );

  const renderTemplatesList = () => {
    if (loading) {
      return (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-gray-600">Loading templates...</span>
        </div>
      );
    }

    if (templates.length === 0) {
      return (
        <div className="text-center py-12">
          <Award className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No templates found</h3>
          <p className="text-gray-600">Try adjusting your search criteria or filters.</p>
        </div>
      );
    }

    return (
      <div className={`grid gap-6 ${
        viewMode === 'grid' 
          ? 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3' 
          : 'grid-cols-1'
      }`}>
        {templates.map(renderTemplateCard)}
      </div>
    );
  };

  const renderTemplateDetail = () => {
    if (!selectedTemplate) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg max-w-4xl max-h-[90vh] overflow-y-auto m-4">
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-start justify-between">
              <div>
                <h2 className="text-2xl font-bold text-gray-900 mb-2">
                  {selectedTemplate.name}
                </h2>
                <p className="text-gray-600 mb-4">{selectedTemplate.description}</p>
                
                <div className="flex items-center gap-4 text-sm text-gray-500">
                  <div className="flex items-center gap-1">
                    <Star className="h-4 w-4 fill-current text-yellow-400" />
                    <span>{selectedTemplate.rating.toFixed(1)}</span>
                    <span>({selectedTemplate.rating_count} reviews)</span>
                  </div>
                  
                  <div className="flex items-center gap-1">
                    <Download className="h-4 w-4" />
                    <span>{selectedTemplate.download_count} downloads</span>
                  </div>
                  
                  <div className="flex items-center gap-1">
                    <Clock className="h-4 w-4" />
                    <span>Updated {new Date(selectedTemplate.updated_at).toLocaleDateString()}</span>
                  </div>
                </div>
              </div>
              
              <button
                onClick={() => setSelectedTemplate(null)}
                className="text-gray-400 hover:text-gray-600"
              >
                ×
              </button>
            </div>
          </div>
          
          <div className="p-6">
            {selectedTemplate.documentation && (
              <div className="mb-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">Documentation</h3>
                <div className="prose prose-sm max-w-none">
                  {selectedTemplate.documentation}
                </div>
              </div>
            )}
            
            <div className="flex items-center justify-between pt-4 border-t border-gray-200">
              <div className="flex items-center gap-4">
                {/* Rating component */}
                <div className="flex items-center gap-1">
                  {[1, 2, 3, 4, 5].map(rating => (
                    <button
                      key={rating}
                      onClick={() => handleRateTemplate(selectedTemplate.id, rating)}
                      className="text-gray-300 hover:text-yellow-400"
                    >
                      <Star className="h-5 w-5 fill-current" />
                    </button>
                  ))}
                </div>
              </div>
              
              <button
                onClick={() => handleInstall(selectedTemplate)}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
              >
                <Download className="h-4 w-4" />
                Install Template
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="h-screen flex flex-col">
      {renderHeader()}
      {renderSearchAndFilters()}
      
      <div className="flex-1 overflow-y-auto p-6">
        {renderTemplatesList()}
      </div>
      
      {renderTemplateDetail()}
    </div>
  );
};