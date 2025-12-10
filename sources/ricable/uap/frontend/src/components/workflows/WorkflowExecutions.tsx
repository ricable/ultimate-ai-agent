/**
 * Workflow Executions Component
 * 
 * View and monitor workflow execution history with detailed logs,
 * performance metrics, and execution management.
 */

import React, { useState, useEffect } from 'react';
import {
  ArrowLeft,
  Play,
  Pause,
  RotateCcw,
  Download,
  Eye,
  AlertCircle,
  CheckCircle,
  Clock,
  X,
  Filter,
  Search,
  Calendar,
  BarChart
} from 'lucide-react';

interface WorkflowExecution {
  id: string;
  execution_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'timeout';
  started_at: string;
  completed_at?: string;
  duration_ms?: number;
  input_data: any;
  output_data?: any;
  error_message?: string;
  error_details?: any;
  trigger_type: string;
  triggered_by: string;
  trigger_data?: any;
  retry_count: number;
  step_executions?: any[];
}

interface Workflow {
  id: string;
  name: string;
  description: string;
  execution_count: number;
  success_count: number;
  failure_count: number;
  avg_duration_ms: number;
}

interface WorkflowExecutionsProps {
  workflow: Workflow;
  onBack: () => void;
}

export const WorkflowExecutions: React.FC<WorkflowExecutionsProps> = ({
  workflow,
  onBack
}) => {
  const [executions, setExecutions] = useState<WorkflowExecution[]>([]);
  const [selectedExecution, setSelectedExecution] = useState<WorkflowExecution | null>(null);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({
    status: '',
    search: '',
    dateRange: '7d'
  });
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  useEffect(() => {
    loadExecutions();
  }, [workflow.id, filters, currentPage]);

  const loadExecutions = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams({
        skip: ((currentPage - 1) * 20).toString(),
        limit: '20',
        ...(filters.status && { status: filters.status }),
        ...(filters.search && { search: filters.search })
      });

      const response = await fetch(`/api/workflows/${workflow.id}/executions?${params}`);
      const data = await response.json();
      
      setExecutions(data.executions || []);
      setTotalPages(data.page_count || 1);
    } catch (error) {
      console.error('Failed to load executions:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRetryExecution = async (execution: WorkflowExecution) => {
    try {
      const response = await fetch(`/api/workflows/${workflow.id}/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input_data: execution.input_data,
          trigger_data: { ...execution.trigger_data, retry_of: execution.id }
        })
      });

      if (response.ok) {
        loadExecutions(); // Refresh list
      }
    } catch (error) {
      console.error('Failed to retry execution:', error);
    }
  };

  const handleCancelExecution = async (execution: WorkflowExecution) => {
    try {
      const response = await fetch(`/api/workflows/executions/${execution.id}/cancel`, {
        method: 'POST'
      });

      if (response.ok) {
        loadExecutions(); // Refresh list
      }
    } catch (error) {
      console.error('Failed to cancel execution:', error);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case 'failed':
        return <AlertCircle className="h-5 w-5 text-red-500" />;
      case 'running':
        return <Clock className="h-5 w-5 text-blue-500 animate-spin" />;
      case 'cancelled':
        return <X className="h-5 w-5 text-gray-500" />;
      default:
        return <Clock className="h-5 w-5 text-yellow-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      case 'running':
        return 'bg-blue-100 text-blue-800';
      case 'cancelled':
        return 'bg-gray-100 text-gray-800';
      default:
        return 'bg-yellow-100 text-yellow-800';
    }
  };

  const formatDuration = (ms?: number) => {
    if (!ms) return '-';
    
    if (ms < 1000) {
      return `${ms}ms`;
    } else if (ms < 60000) {
      return `${(ms / 1000).toFixed(1)}s`;
    } else {
      return `${(ms / 60000).toFixed(1)}m`;
    }
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
            <h1 className="text-2xl font-bold text-gray-900">Workflow Executions</h1>
            <p className="text-gray-600">{workflow.name}</p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <button
            onClick={loadExecutions}
            className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 flex items-center gap-2"
          >
            <RotateCcw className="h-4 w-4" />
            Refresh
          </button>
        </div>
      </div>
      
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mt-6">
        <div className="bg-blue-50 rounded-lg p-4">
          <div className="flex items-center">
            <div className="p-2 bg-blue-100 rounded-lg">
              <BarChart className="h-6 w-6 text-blue-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-blue-600">Total Executions</p>
              <p className="text-2xl font-bold text-blue-900">{workflow.execution_count}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-green-50 rounded-lg p-4">
          <div className="flex items-center">
            <div className="p-2 bg-green-100 rounded-lg">
              <CheckCircle className="h-6 w-6 text-green-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-green-600">Successful</p>
              <p className="text-2xl font-bold text-green-900">{workflow.success_count}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-red-50 rounded-lg p-4">
          <div className="flex items-center">
            <div className="p-2 bg-red-100 rounded-lg">
              <AlertCircle className="h-6 w-6 text-red-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-red-600">Failed</p>
              <p className="text-2xl font-bold text-red-900">{workflow.failure_count}</p>
            </div>
          </div>
        </div>
        
        <div className="bg-purple-50 rounded-lg p-4">
          <div className="flex items-center">
            <div className="p-2 bg-purple-100 rounded-lg">
              <Clock className="h-6 w-6 text-purple-600" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-purple-600">Avg Duration</p>
              <p className="text-2xl font-bold text-purple-900">{formatDuration(workflow.avg_duration_ms)}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderFilters = () => (
    <div className="bg-white border-b border-gray-200 px-6 py-4">
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
          <input
            type="text"
            placeholder="Search executions..."
            value={filters.search}
            onChange={(e) => setFilters(prev => ({ ...prev, search: e.target.value }))}
            className="pl-10 pr-4 py-2 w-full border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
        
        <select
          value={filters.status}
          onChange={(e) => setFilters(prev => ({ ...prev, status: e.target.value }))}
          className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          <option value="">All Statuses</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
          <option value="running">Running</option>
          <option value="cancelled">Cancelled</option>
          <option value="pending">Pending</option>
        </select>
        
        <select
          value={filters.dateRange}
          onChange={(e) => setFilters(prev => ({ ...prev, dateRange: e.target.value }))}
          className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          <option value="1d">Last 24 hours</option>
          <option value="7d">Last 7 days</option>
          <option value="30d">Last 30 days</option>
          <option value="90d">Last 90 days</option>
        </select>
      </div>
    </div>
  );

  const renderExecutionsList = () => (
    <div className="bg-white shadow overflow-hidden">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Execution
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Status
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Started
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Duration
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Trigger
            </th>
            <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
              Actions
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {executions.map((execution) => (
            <tr key={execution.id} className="hover:bg-gray-50">
              <td className="px-6 py-4 whitespace-nowrap">
                <div>
                  <div className="text-sm font-medium text-gray-900">
                    {execution.execution_id}
                  </div>
                  <div className="text-sm text-gray-500">
                    ID: {execution.id.substring(0, 8)}...
                  </div>
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="flex items-center gap-2">
                  {getStatusIcon(execution.status)}
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(execution.status)}`}>
                    {execution.status}
                  </span>
                </div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                {new Date(execution.started_at).toLocaleString()}
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                {formatDuration(execution.duration_ms)}
              </td>
              <td className="px-6 py-4 whitespace-nowrap">
                <div className="text-sm text-gray-900">{execution.trigger_type}</div>
                <div className="text-sm text-gray-500">by {execution.triggered_by}</div>
              </td>
              <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                <div className="flex items-center justify-end gap-2">
                  <button
                    onClick={() => setSelectedExecution(execution)}
                    className="text-blue-600 hover:text-blue-900"
                  >
                    <Eye className="h-4 w-4" />
                  </button>
                  
                  {execution.status === 'failed' && (
                    <button
                      onClick={() => handleRetryExecution(execution)}
                      className="text-green-600 hover:text-green-900"
                      title="Retry execution"
                    >
                      <RotateCcw className="h-4 w-4" />
                    </button>
                  )}
                  
                  {execution.status === 'running' && (
                    <button
                      onClick={() => handleCancelExecution(execution)}
                      className="text-red-600 hover:text-red-900"
                      title="Cancel execution"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  )}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      
      {loading && (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-gray-600">Loading executions...</span>
        </div>
      )}
      
      {!loading && executions.length === 0 && (
        <div className="text-center py-12">
          <BarChart className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No executions found</h3>
          <p className="text-gray-600">This workflow hasn't been executed yet.</p>
        </div>
      )}
    </div>
  );

  const renderPagination = () => {
    if (totalPages <= 1) return null;
    
    return (
      <div className="bg-white px-4 py-3 flex items-center justify-between border-t border-gray-200">
        <div className="flex-1 flex justify-between sm:hidden">
          <button
            onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
            disabled={currentPage === 1}
            className="relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50"
          >
            Previous
          </button>
          <button
            onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
            disabled={currentPage === totalPages}
            className="ml-3 relative inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50"
          >
            Next
          </button>
        </div>
        <div className="hidden sm:flex-1 sm:flex sm:items-center sm:justify-between">
          <div>
            <p className="text-sm text-gray-700">
              Page <span className="font-medium">{currentPage}</span> of{' '}
              <span className="font-medium">{totalPages}</span>
            </p>
          </div>
          <div>
            <nav className="relative z-0 inline-flex rounded-md shadow-sm -space-x-px">
              <button
                onClick={() => setCurrentPage(prev => Math.max(1, prev - 1))}
                disabled={currentPage === 1}
                className="relative inline-flex items-center px-2 py-2 rounded-l-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50"
              >
                Previous
              </button>
              <button
                onClick={() => setCurrentPage(prev => Math.min(totalPages, prev + 1))}
                disabled={currentPage === totalPages}
                className="relative inline-flex items-center px-2 py-2 rounded-r-md border border-gray-300 bg-white text-sm font-medium text-gray-500 hover:bg-gray-50 disabled:opacity-50"
              >
                Next
              </button>
            </nav>
          </div>
        </div>
      </div>
    );
  };

  const renderExecutionDetail = () => {
    if (!selectedExecution) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg max-w-4xl max-h-[90vh] overflow-y-auto m-4 w-full">
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-start justify-between">
              <div>
                <h2 className="text-2xl font-bold text-gray-900 mb-2">
                  Execution Details
                </h2>
                <p className="text-gray-600">{selectedExecution.execution_id}</p>
              </div>
              
              <button
                onClick={() => setSelectedExecution(null)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="h-6 w-6" />
              </button>
            </div>
          </div>
          
          <div className="p-6 space-y-6">
            <div className="grid grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-3">Execution Info</h3>
                <dl className="space-y-2">
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Status</dt>
                    <dd className="flex items-center gap-2">
                      {getStatusIcon(selectedExecution.status)}
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(selectedExecution.status)}`}>
                        {selectedExecution.status}
                      </span>
                    </dd>
                  </div>
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Started</dt>
                    <dd className="text-sm text-gray-900">{new Date(selectedExecution.started_at).toLocaleString()}</dd>
                  </div>
                  {selectedExecution.completed_at && (
                    <div>
                      <dt className="text-sm font-medium text-gray-500">Completed</dt>
                      <dd className="text-sm text-gray-900">{new Date(selectedExecution.completed_at).toLocaleString()}</dd>
                    </div>
                  )}
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Duration</dt>
                    <dd className="text-sm text-gray-900">{formatDuration(selectedExecution.duration_ms)}</dd>
                  </div>
                  <div>
                    <dt className="text-sm font-medium text-gray-500">Triggered By</dt>
                    <dd className="text-sm text-gray-900">{selectedExecution.triggered_by}</dd>
                  </div>
                </dl>
              </div>
              
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-3">Input Data</h3>
                <pre className="text-sm bg-gray-100 p-3 rounded-lg overflow-auto max-h-48">
                  {JSON.stringify(selectedExecution.input_data, null, 2)}
                </pre>
              </div>
            </div>
            
            {selectedExecution.output_data && (
              <div>
                <h3 className="text-lg font-medium text-gray-900 mb-3">Output Data</h3>
                <pre className="text-sm bg-gray-100 p-3 rounded-lg overflow-auto max-h-48">
                  {JSON.stringify(selectedExecution.output_data, null, 2)}
                </pre>
              </div>
            )}
            
            {selectedExecution.error_message && (
              <div>
                <h3 className="text-lg font-medium text-red-900 mb-3">Error Details</h3>
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <p className="text-sm text-red-800 mb-2">{selectedExecution.error_message}</p>
                  {selectedExecution.error_details && (
                    <pre className="text-xs text-red-700 overflow-auto max-h-32">
                      {JSON.stringify(selectedExecution.error_details, null, 2)}
                    </pre>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="h-screen flex flex-col">
      {renderHeader()}
      {renderFilters()}
      
      <div className="flex-1 overflow-y-auto">
        {renderExecutionsList()}
        {renderPagination()}
      </div>
      
      {renderExecutionDetail()}
    </div>
  );
};