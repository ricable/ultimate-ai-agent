/**
 * useWorkflows Hook
 * 
 * Custom React hook for managing workflow operations including
 * CRUD operations, execution, and statistics.
 */

import { useState, useCallback } from 'react';

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

interface FilterOptions {
  status: string;
  search: string;
  tags: string[];
  sortBy: 'name' | 'created_at' | 'execution_count' | 'success_rate';
  sortOrder: 'asc' | 'desc';
}

interface WorkflowStats {
  total_workflows: number;
  active_workflows: number;
  total_executions: number;
  scheduled_workflows: number;
  avg_success_rate: number;
}

interface UseWorkflowsReturn {
  workflows: Workflow[];
  loading: boolean;
  error: string | null;
  stats: WorkflowStats | null;
  
  // CRUD operations
  createWorkflow: (workflowData: any) => Promise<void>;
  updateWorkflow: (id: string, workflowData: any) => Promise<void>;
  deleteWorkflow: (id: string) => Promise<void>;
  
  // Execution
  executeWorkflow: (id: string, inputData: any) => Promise<any>;
  
  // Data loading
  loadWorkflows: (filters?: Partial<FilterOptions>) => Promise<void>;
  loadWorkflowStats: () => Promise<void>;
}

export const useWorkflows = (): UseWorkflowsReturn => {
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<WorkflowStats | null>(null);

  const handleError = (error: any, defaultMessage: string) => {
    const message = error?.message || error?.detail || defaultMessage;
    setError(message);
    console.error(defaultMessage, error);
  };

  const loadWorkflows = useCallback(async (filters: Partial<FilterOptions> = {}) => {
    setLoading(true);
    setError(null);
    
    try {
      const params = new URLSearchParams();
      
      if (filters.status) params.append('status', filters.status);
      if (filters.search) params.append('search', filters.search);
      if (filters.sortBy) params.append('sort_by', filters.sortBy);
      if (filters.sortOrder) params.append('sort_order', filters.sortOrder);
      
      const response = await fetch(`/api/workflows?${params}`);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setWorkflows(data.workflows || []);
    } catch (error) {
      handleError(error, 'Failed to load workflows');
    } finally {
      setLoading(false);
    }
  }, []);

  const loadWorkflowStats = useCallback(async () => {
    try {
      const response = await fetch('/api/workflows/system/status');
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Calculate stats from the response
      const totalWorkflows = workflows.length;
      const activeWorkflows = workflows.filter(w => w.status === 'active').length;
      const totalExecutions = workflows.reduce((sum, w) => sum + w.execution_count, 0);
      const scheduledWorkflows = workflows.filter(w => w.triggers?.some(t => t.trigger_type === 'schedule')).length;
      const avgSuccessRate = totalExecutions > 0 
        ? workflows.reduce((sum, w) => sum + (w.execution_count > 0 ? (w.success_count / w.execution_count) * 100 : 0), 0) / workflows.length
        : 0;

      setStats({
        total_workflows: totalWorkflows,
        active_workflows: activeWorkflows,
        total_executions: totalExecutions,
        scheduled_workflows: scheduledWorkflows,
        avg_success_rate: Math.round(avgSuccessRate)
      });
    } catch (error) {
      handleError(error, 'Failed to load workflow statistics');
    }
  }, [workflows]);

  const createWorkflow = useCallback(async (workflowData: any) => {
    try {
      const response = await fetch('/api/workflows', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(workflowData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      
      // Refresh workflows list
      await loadWorkflows();
      
      return result;
    } catch (error) {
      handleError(error, 'Failed to create workflow');
      throw error;
    }
  }, [loadWorkflows]);

  const updateWorkflow = useCallback(async (id: string, workflowData: any) => {
    try {
      const response = await fetch(`/api/workflows/${id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(workflowData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      
      // Update local state
      setWorkflows(prev => prev.map(w => 
        w.id === id ? { ...w, ...workflowData, updated_at: new Date().toISOString() } : w
      ));
      
      return result;
    } catch (error) {
      handleError(error, 'Failed to update workflow');
      throw error;
    }
  }, []);

  const deleteWorkflow = useCallback(async (id: string) => {
    try {
      const response = await fetch(`/api/workflows/${id}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      // Remove from local state
      setWorkflows(prev => prev.filter(w => w.id !== id));
      
      return true;
    } catch (error) {
      handleError(error, 'Failed to delete workflow');
      throw error;
    }
  }, []);

  const executeWorkflow = useCallback(async (id: string, inputData: any = {}) => {
    try {
      const response = await fetch(`/api/workflows/${id}/execute`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          input_data: inputData,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const result = await response.json();
      
      // Update execution count in local state
      setWorkflows(prev => prev.map(w => 
        w.id === id 
          ? { 
              ...w, 
              execution_count: w.execution_count + 1,
              ...(result.success && { success_count: w.success_count + 1 }),
              ...(!result.success && { failure_count: w.failure_count + 1 })
            }
          : w
      ));
      
      return result;
    } catch (error) {
      handleError(error, 'Failed to execute workflow');
      throw error;
    }
  }, []);

  // Additional utility functions
  const getWorkflowById = useCallback((id: string): Workflow | undefined => {
    return workflows.find(w => w.id === id);
  }, [workflows]);

  const getWorkflowsByStatus = useCallback((status: Workflow['status']): Workflow[] => {
    return workflows.filter(w => w.status === status);
  }, [workflows]);

  const getWorkflowsByTag = useCallback((tag: string): Workflow[] => {
    return workflows.filter(w => w.tags?.includes(tag));
  }, [workflows]);

  const searchWorkflows = useCallback((query: string): Workflow[] => {
    const lowerQuery = query.toLowerCase();
    return workflows.filter(w => 
      w.name.toLowerCase().includes(lowerQuery) ||
      w.description?.toLowerCase().includes(lowerQuery) ||
      w.tags?.some(tag => tag.toLowerCase().includes(lowerQuery))
    );
  }, [workflows]);

  return {
    workflows,
    loading,
    error,
    stats,
    
    // CRUD operations
    createWorkflow,
    updateWorkflow,
    deleteWorkflow,
    
    // Execution
    executeWorkflow,
    
    // Data loading
    loadWorkflows,
    loadWorkflowStats,
    
    // Utility functions (not part of the interface but available)
    getWorkflowById,
    getWorkflowsByStatus,
    getWorkflowsByTag,
    searchWorkflows,
  } as UseWorkflowsReturn & {
    getWorkflowById: (id: string) => Workflow | undefined;
    getWorkflowsByStatus: (status: Workflow['status']) => Workflow[];
    getWorkflowsByTag: (tag: string) => Workflow[];
    searchWorkflows: (query: string) => Workflow[];
  };
};