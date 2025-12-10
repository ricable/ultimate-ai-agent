/**
 * Workflow Dashboard
 * 
 * Main dashboard for workflow management with create, edit, execute,
 * and monitor capabilities.
 */

import React, { useState, useEffect } from 'react';
import { Plus, Play, Pause, Settings, Calendar, Zap, Search, Filter } from 'lucide-react';
import { WorkflowCard } from './WorkflowCard';
import { WorkflowEditor } from './WorkflowEditor';
import { WorkflowExecutions } from './WorkflowExecutions';
import { WorkflowScheduler } from './WorkflowScheduler';
import { WorkflowMarketplace } from './WorkflowMarketplace';
import { useWorkflows } from '../../hooks/useWorkflows';

interface Workflow {
  id: string;
  name: string;
  description: string;
  status: 'draft' | 'active' | 'paused' | 'archived';
  definition: any;
  variables: any;
  tags: string[];
  created_at: string;
  updated_at: string;
  execution_count: number;
  success_count: number;
  failure_count: number;
  avg_duration_ms: number;
  steps: any[];
  triggers: any[];
}

type ViewMode = 'list' | 'create' | 'edit' | 'executions' | 'schedule' | 'marketplace';

interface FilterOptions {
  status: string;
  search: string;
  tags: string[];
  sortBy: 'name' | 'created_at' | 'execution_count' | 'success_rate';
  sortOrder: 'asc' | 'desc';
}

export const WorkflowDashboard: React.FC = () => {
  const [viewMode, setViewMode] = useState<ViewMode>('list');
  const [selectedWorkflow, setSelectedWorkflow] = useState<Workflow | null>(null);
  const [filters, setFilters] = useState<FilterOptions>({
    status: '',
    search: '',
    tags: [],
    sortBy: 'created_at',
    sortOrder: 'desc'
  });
  const [showFilters, setShowFilters] = useState(false);

  const {
    workflows,
    loading,
    error,
    createWorkflow,
    updateWorkflow,
    deleteWorkflow,
    executeWorkflow,
    loadWorkflows,
    stats
  } = useWorkflows();

  useEffect(() => {
    loadWorkflows(filters);
  }, [filters, loadWorkflows]);

  const handleCreateWorkflow = () => {
    setSelectedWorkflow(null);
    setViewMode('create');
  };

  const handleEditWorkflow = (workflow: Workflow) => {
    setSelectedWorkflow(workflow);
    setViewMode('edit');
  };

  const handleViewExecutions = (workflow: Workflow) => {
    setSelectedWorkflow(workflow);
    setViewMode('executions');
  };

  const handleScheduleWorkflow = (workflow: Workflow) => {
    setSelectedWorkflow(workflow);
    setViewMode('schedule');
  };

  const handleExecuteWorkflow = async (workflow: Workflow) => {
    try {
      await executeWorkflow(workflow.id, {});
      // Show success notification
    } catch (error) {
      // Show error notification
      console.error('Failed to execute workflow:', error);
    }
  };

  const handleSaveWorkflow = async (workflowData: any) => {
    try {
      if (selectedWorkflow) {
        await updateWorkflow(selectedWorkflow.id, workflowData);
      } else {
        await createWorkflow(workflowData);
      }
      setViewMode('list');
      loadWorkflows(filters);
    } catch (error) {
      console.error('Failed to save workflow:', error);
    }
  };

  const handleDeleteWorkflow = async (workflowId: string) => {
    if (window.confirm('Are you sure you want to delete this workflow?')) {
      try {
        await deleteWorkflow(workflowId);
        loadWorkflows(filters);
      } catch (error) {
        console.error('Failed to delete workflow:', error);
      }
    }
  };

  const handleUpdateFilters = (newFilters: Partial<FilterOptions>) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
  };

  const filteredWorkflows = workflows.filter(workflow => {
    if (filters.status && workflow.status !== filters.status) return false;
    if (filters.search && !workflow.name.toLowerCase().includes(filters.search.toLowerCase()) &&
        !workflow.description?.toLowerCase().includes(filters.search.toLowerCase())) return false;
    if (filters.tags.length > 0 && !filters.tags.some(tag => workflow.tags?.includes(tag))) return false;
    return true;
  }).sort((a, b) => {
    const { sortBy, sortOrder } = filters;
    let comparison = 0;
    
    switch (sortBy) {
      case 'name':
        comparison = a.name.localeCompare(b.name);
        break;
      case 'created_at':
        comparison = new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
        break;
      case 'execution_count':
        comparison = a.execution_count - b.execution_count;
        break;
      case 'success_rate':
        const aRate = a.execution_count > 0 ? a.success_count / a.execution_count : 0;
        const bRate = b.execution_count > 0 ? b.success_count / b.execution_count : 0;
        comparison = aRate - bRate;
        break;
    }
    
    return sortOrder === 'asc' ? comparison : -comparison;
  });

  const renderStatsCards = () => (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center">
          <div className="p-2 bg-blue-100 rounded-lg">
            <Settings className="h-6 w-6 text-blue-600" />
          </div>
          <div className="ml-4">
            <p className="text-sm font-medium text-gray-600">Total Workflows</p>
            <p className="text-2xl font-bold text-gray-900">{stats?.total_workflows || 0}</p>
          </div>
        </div>
      </div>
      
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center">
          <div className="p-2 bg-green-100 rounded-lg">
            <Play className="h-6 w-6 text-green-600" />
          </div>
          <div className="ml-4">
            <p className="text-sm font-medium text-gray-600">Active Workflows</p>
            <p className="text-2xl font-bold text-gray-900">{stats?.active_workflows || 0}</p>
          </div>
        </div>
      </div>
      
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center">
          <div className="p-2 bg-purple-100 rounded-lg">
            <Zap className="h-6 w-6 text-purple-600" />
          </div>
          <div className="ml-4">
            <p className="text-sm font-medium text-gray-600">Total Executions</p>
            <p className="text-2xl font-bold text-gray-900">{stats?.total_executions || 0}</p>
          </div>
        </div>
      </div>
      
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center">
          <div className="p-2 bg-yellow-100 rounded-lg">
            <Calendar className="h-6 w-6 text-yellow-600" />
          </div>
          <div className="ml-4">
            <p className="text-sm font-medium text-gray-600">Scheduled</p>
            <p className="text-2xl font-bold text-gray-900">{stats?.scheduled_workflows || 0}</p>
          </div>
        </div>
      </div>
    </div>
  );

  const renderToolbar = () => (
    <div className="flex flex-col sm:flex-row gap-4 mb-6">
      <div className="flex-1 flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
          <input
            type="text"
            placeholder="Search workflows..."
            value={filters.search}
            onChange={(e) => handleUpdateFilters({ search: e.target.value })}
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
      
      <div className="flex gap-2">
        <button
          onClick={() => setViewMode('marketplace')}
          className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
        >
          Marketplace
        </button>
        <button
          onClick={handleCreateWorkflow}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
        >
          <Plus className="h-4 w-4" />
          New Workflow
        </button>
      </div>
    </div>
  );

  const renderFilters = () => {
    if (!showFilters) return null;
    
    return (
      <div className="bg-gray-50 p-4 rounded-lg mb-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Status</label>
            <select
              value={filters.status}
              onChange={(e) => handleUpdateFilters({ status: e.target.value })}
              className="w-full border border-gray-300 rounded-lg px-3 py-2"
            >
              <option value="">All Statuses</option>
              <option value="draft">Draft</option>
              <option value="active">Active</option>
              <option value="paused">Paused</option>
              <option value="archived">Archived</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Sort By</label>
            <select
              value={filters.sortBy}
              onChange={(e) => handleUpdateFilters({ sortBy: e.target.value as any })}
              className="w-full border border-gray-300 rounded-lg px-3 py-2"
            >
              <option value="created_at">Created Date</option>
              <option value="name">Name</option>
              <option value="execution_count">Execution Count</option>
              <option value="success_rate">Success Rate</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Order</label>
            <select
              value={filters.sortOrder}
              onChange={(e) => handleUpdateFilters({ sortOrder: e.target.value as any })}
              className="w-full border border-gray-300 rounded-lg px-3 py-2"
            >
              <option value="desc">Descending</option>
              <option value="asc">Ascending</option>
            </select>
          </div>
          
          <div className="flex items-end">
            <button
              onClick={() => setFilters({
                status: '',
                search: '',
                tags: [],
                sortBy: 'created_at',
                sortOrder: 'desc'
              })}
              className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Clear Filters
            </button>
          </div>
        </div>
      </div>
    );
  };

  const renderWorkflowList = () => (
    <div className="space-y-4">
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-gray-600">Loading workflows...</span>
        </div>
      ) : error ? (
        <div className="text-center py-12">
          <p className="text-red-600 mb-4">{error}</p>
          <button
            onClick={() => loadWorkflows(filters)}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      ) : filteredWorkflows.length === 0 ? (
        <div className="text-center py-12">
          <Settings className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No workflows found</h3>
          <p className="text-gray-600 mb-4">
            {filters.search || filters.status || filters.tags.length > 0
              ? 'No workflows match your current filters.'
              : 'Get started by creating your first workflow.'
            }
          </p>
          {!filters.search && !filters.status && filters.tags.length === 0 && (
            <button
              onClick={handleCreateWorkflow}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Create Workflow
            </button>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {filteredWorkflows.map((workflow) => (
            <WorkflowCard
              key={workflow.id}
              workflow={workflow}
              onEdit={handleEditWorkflow}
              onExecute={handleExecuteWorkflow}
              onViewExecutions={handleViewExecutions}
              onSchedule={handleScheduleWorkflow}
              onDelete={handleDeleteWorkflow}
            />
          ))}
        </div>
      )}
    </div>
  );

  // Render different views based on current mode
  if (viewMode === 'create' || viewMode === 'edit') {
    return (
      <WorkflowEditor
        workflow={selectedWorkflow}
        onSave={handleSaveWorkflow}
        onCancel={() => setViewMode('list')}
      />
    );
  }

  if (viewMode === 'executions' && selectedWorkflow) {
    return (
      <WorkflowExecutions
        workflow={selectedWorkflow}
        onBack={() => setViewMode('list')}
      />
    );
  }

  if (viewMode === 'schedule' && selectedWorkflow) {
    return (
      <WorkflowScheduler
        workflow={selectedWorkflow}
        onBack={() => setViewMode('list')}
      />
    );
  }

  if (viewMode === 'marketplace') {
    return (
      <WorkflowMarketplace
        onBack={() => setViewMode('list')}
        onInstall={(templateId, workflowName) => {
          // Handle template installation
          setViewMode('list');
          loadWorkflows(filters);
        }}
      />
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Workflow Automation</h1>
        <p className="mt-2 text-gray-600">
          Create, manage, and automate your business processes with powerful workflows.
        </p>
      </div>

      {renderStatsCards()}
      {renderToolbar()}
      {renderFilters()}
      {renderWorkflowList()}
    </div>
  );
};