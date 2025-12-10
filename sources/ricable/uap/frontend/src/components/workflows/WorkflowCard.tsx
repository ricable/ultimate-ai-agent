/**
 * Workflow Card Component
 * 
 * Individual workflow card displaying workflow information and actions
 * in the workflow dashboard.
 */

import React, { useState } from 'react';
import {
  Play,
  Pause,
  Edit,
  Trash2,
  Calendar,
  BarChart3,
  Clock,
  CheckCircle,
  XCircle,
  MoreVertical,
  Copy,
  Download,
  Share
} from 'lucide-react';

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

interface WorkflowCardProps {
  workflow: Workflow;
  onEdit: (workflow: Workflow) => void;
  onExecute: (workflow: Workflow) => void;
  onViewExecutions: (workflow: Workflow) => void;
  onSchedule: (workflow: Workflow) => void;
  onDelete: (workflowId: string) => void;
  onDuplicate?: (workflow: Workflow) => void;
  onExport?: (workflow: Workflow) => void;
  onShare?: (workflow: Workflow) => void;
}

export const WorkflowCard: React.FC<WorkflowCardProps> = ({
  workflow,
  onEdit,
  onExecute,
  onViewExecutions,
  onSchedule,
  onDelete,
  onDuplicate,
  onExport,
  onShare
}) => {
  const [showDropdown, setShowDropdown] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'bg-green-100 text-green-800';
      case 'paused':
        return 'bg-yellow-100 text-yellow-800';
      case 'draft':
        return 'bg-gray-100 text-gray-800';
      case 'archived':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <Play className="h-3 w-3" />;
      case 'paused':
        return <Pause className="h-3 w-3" />;
      case 'draft':
        return <Edit className="h-3 w-3" />;
      case 'archived':
        return <XCircle className="h-3 w-3" />;
      default:
        return <Edit className="h-3 w-3" />;
    }
  };

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getSuccessRate = () => {
    if (workflow.execution_count === 0) return 0;
    return Math.round((workflow.success_count / workflow.execution_count) * 100);
  };

  const handleExecute = async () => {
    if (isExecuting) return;
    
    setIsExecuting(true);
    try {
      await onExecute(workflow);
    } finally {
      setIsExecuting(false);
    }
  };

  const handleDropdownAction = (action: string) => {
    setShowDropdown(false);
    
    switch (action) {
      case 'duplicate':
        onDuplicate?.(workflow);
        break;
      case 'export':
        onExport?.(workflow);
        break;
      case 'share':
        onShare?.(workflow);
        break;
      case 'delete':
        onDelete(workflow.id);
        break;
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow duration-200 overflow-hidden">
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-start justify-between">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-3 mb-2">
              <h3 className="text-lg font-semibold text-gray-900 truncate">
                {workflow.name}
              </h3>
              <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(workflow.status)}`}>
                {getStatusIcon(workflow.status)}
                {workflow.status}
              </span>
            </div>
            
            {workflow.description && (
              <p className="text-sm text-gray-600 line-clamp-2 mb-3">
                {workflow.description}
              </p>
            )}
            
            {workflow.tags && workflow.tags.length > 0 && (
              <div className="flex flex-wrap gap-1 mb-3">
                {workflow.tags.slice(0, 3).map((tag, index) => (
                  <span
                    key={index}
                    className="inline-flex items-center px-2 py-1 rounded-md text-xs font-medium bg-blue-100 text-blue-800"
                  >
                    {tag}
                  </span>
                ))}
                {workflow.tags.length > 3 && (
                  <span className="text-xs text-gray-500">
                    +{workflow.tags.length - 3} more
                  </span>
                )}
              </div>
            )}
          </div>
          
          <div className="relative ml-3">
            <button
              onClick={() => setShowDropdown(!showDropdown)}
              className="p-1 rounded-md hover:bg-gray-100"
            >
              <MoreVertical className="h-4 w-4 text-gray-400" />
            </button>
            
            {showDropdown && (
              <div className="absolute right-0 top-8 w-48 bg-white rounded-md shadow-lg border border-gray-200 z-10">
                <div className="py-1">
                  {onDuplicate && (
                    <button
                      onClick={() => handleDropdownAction('duplicate')}
                      className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
                    >
                      <Copy className="h-4 w-4" />
                      Duplicate
                    </button>
                  )}
                  {onExport && (
                    <button
                      onClick={() => handleDropdownAction('export')}
                      className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
                    >
                      <Download className="h-4 w-4" />
                      Export
                    </button>
                  )}
                  {onShare && (
                    <button
                      onClick={() => handleDropdownAction('share')}
                      className="flex items-center gap-2 w-full px-4 py-2 text-sm text-gray-700 hover:bg-gray-50"
                    >
                      <Share className="h-4 w-4" />
                      Share
                    </button>
                  )}
                  <div className="border-t border-gray-100"></div>
                  <button
                    onClick={() => handleDropdownAction('delete')}
                    className="flex items-center gap-2 w-full px-4 py-2 text-sm text-red-700 hover:bg-red-50"
                  >
                    <Trash2 className="h-4 w-4" />
                    Delete
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Stats */}
      <div className="p-6 space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">
              {workflow.execution_count}
            </div>
            <div className="text-sm text-gray-600">Executions</div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {getSuccessRate()}%
            </div>
            <div className="text-sm text-gray-600">Success Rate</div>
          </div>
        </div>
        
        {workflow.execution_count > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-600">Performance</span>
              <span className="text-gray-900">
                Avg: {formatDuration(workflow.avg_duration_ms)}
              </span>
            </div>
            
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-green-500 h-2 rounded-full"
                style={{ width: `${getSuccessRate()}%` }}
              ></div>
            </div>
            
            <div className="flex justify-between text-xs text-gray-500">
              <span>{workflow.success_count} successful</span>
              <span>{workflow.failure_count} failed</span>
            </div>
          </div>
        )}
        
        <div className="flex items-center justify-between text-sm text-gray-500">
          <div className="flex items-center gap-1">
            <Clock className="h-4 w-4" />
            Updated {formatDate(workflow.updated_at)}
          </div>
          <div className="flex items-center gap-1">
            <span>{workflow.steps?.length || 0} steps</span>
          </div>
        </div>
        
        {workflow.triggers && workflow.triggers.length > 0 && (
          <div className="flex items-center gap-2">
            <Calendar className="h-4 w-4 text-gray-400" />
            <span className="text-sm text-gray-600">
              {workflow.triggers.length} trigger{workflow.triggers.length !== 1 ? 's' : ''}
            </span>
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="px-6 py-4 bg-gray-50 border-t border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <button
              onClick={() => onEdit(workflow)}
              className="px-3 py-1.5 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <Edit className="h-3 w-3 mr-1 inline" />
              Edit
            </button>
            
            <button
              onClick={() => onViewExecutions(workflow)}
              className="px-3 py-1.5 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <BarChart3 className="h-3 w-3 mr-1 inline" />
              History
            </button>
            
            <button
              onClick={() => onSchedule(workflow)}
              className="px-3 py-1.5 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <Calendar className="h-3 w-3 mr-1 inline" />
              Schedule
            </button>
          </div>
          
          <button
            onClick={handleExecute}
            disabled={isExecuting || workflow.status !== 'active'}
            className="px-4 py-1.5 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isExecuting ? (
              <>
                <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white"></div>
                Running...
              </>
            ) : (
              <>
                <Play className="h-3 w-3" />
                Execute
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};