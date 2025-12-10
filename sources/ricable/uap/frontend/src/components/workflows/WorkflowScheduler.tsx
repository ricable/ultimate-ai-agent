/**
 * Workflow Scheduler Component
 * 
 * Manage workflow schedules with cron expressions, human-readable patterns,
 * and advanced scheduling options.
 */

import React, { useState, useEffect } from 'react';
import {
  ArrowLeft,
  Plus,
  Edit,
  Trash2,
  Play,
  Pause,
  Clock,
  Calendar,
  Settings,
  AlertCircle,
  CheckCircle,
  Save,
  X
} from 'lucide-react';

interface WorkflowTrigger {
  id: string;
  name: string;
  trigger_type: string;
  schedule: string;
  is_active: boolean;
  execution_count: number;
  last_executed_at?: string;
  next_execution_at?: string;
  config?: any;
}

interface Workflow {
  id: string;
  name: string;
  description: string;
}

interface WorkflowSchedulerProps {
  workflow: Workflow;
  onBack: () => void;
}

interface ScheduleForm {
  name: string;
  schedule: string;
  scheduleType: 'cron' | 'human';
  humanPattern: string;
  timezone: string;
  startDate: string;
  endDate: string;
  maxExecutions: string;
  config: any;
}

const HUMAN_PATTERNS = [
  { value: 'every minute', label: 'Every minute' },
  { value: 'every 5 minutes', label: 'Every 5 minutes' },
  { value: 'every 15 minutes', label: 'Every 15 minutes' },
  { value: 'every 30 minutes', label: 'Every 30 minutes' },
  { value: 'every hour', label: 'Every hour' },
  { value: 'every 6 hours', label: 'Every 6 hours' },
  { value: 'daily', label: 'Daily (midnight)' },
  { value: 'weekdays', label: 'Weekdays (9 AM)' },
  { value: 'weekly', label: 'Weekly (Sunday midnight)' },
  { value: 'monthly', label: 'Monthly (1st, midnight)' },
  { value: 'at 9:00 AM', label: 'Daily at 9:00 AM' },
  { value: 'at 6:00 PM', label: 'Daily at 6:00 PM' },
  { value: 'on weekdays at 9:00 AM', label: 'Weekdays at 9:00 AM' }
];

const TIMEZONES = [
  'UTC',
  'America/New_York',
  'America/Los_Angeles',
  'America/Chicago',
  'Europe/London',
  'Europe/Paris',
  'Asia/Tokyo',
  'Asia/Shanghai',
  'Australia/Sydney'
];

export const WorkflowScheduler: React.FC<WorkflowSchedulerProps> = ({
  workflow,
  onBack
}) => {
  const [schedules, setSchedules] = useState<WorkflowTrigger[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [editingSchedule, setEditingSchedule] = useState<WorkflowTrigger | null>(null);
  const [scheduleForm, setScheduleForm] = useState<ScheduleForm>({
    name: '',
    schedule: '',
    scheduleType: 'human',
    humanPattern: 'daily',
    timezone: 'UTC',
    startDate: '',
    endDate: '',
    maxExecutions: '',
    config: {}
  });
  const [formErrors, setFormErrors] = useState<string[]>([]);

  useEffect(() => {
    loadSchedules();
  }, [workflow.id]);

  const loadSchedules = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/workflows/${workflow.id}/schedules`);
      const data = await response.json();
      setSchedules(data || []);
    } catch (error) {
      console.error('Failed to load schedules:', error);
    } finally {
      setLoading(false);
    }
  };

  const validateForm = (): boolean => {
    const errors: string[] = [];

    if (!scheduleForm.name.trim()) {
      errors.push('Schedule name is required');
    }

    if (scheduleForm.scheduleType === 'cron' && !scheduleForm.schedule.trim()) {
      errors.push('Cron expression is required');
    }

    if (scheduleForm.scheduleType === 'human' && !scheduleForm.humanPattern) {
      errors.push('Schedule pattern is required');
    }

    if (scheduleForm.startDate && scheduleForm.endDate) {
      if (new Date(scheduleForm.startDate) >= new Date(scheduleForm.endDate)) {
        errors.push('End date must be after start date');
      }
    }

    if (scheduleForm.maxExecutions && parseInt(scheduleForm.maxExecutions) <= 0) {
      errors.push('Max executions must be greater than 0');
    }

    setFormErrors(errors);
    return errors.length === 0;
  };

  const handleCreateSchedule = async () => {
    if (!validateForm()) return;

    try {
      const payload = {
        trigger_name: scheduleForm.name,
        schedule: scheduleForm.scheduleType === 'cron' 
          ? scheduleForm.schedule 
          : scheduleForm.humanPattern,
        timezone: scheduleForm.timezone,
        config: {
          ...scheduleForm.config,
          ...(scheduleForm.startDate && { start_date: scheduleForm.startDate }),
          ...(scheduleForm.endDate && { end_date: scheduleForm.endDate }),
          ...(scheduleForm.maxExecutions && { max_executions: parseInt(scheduleForm.maxExecutions) })
        }
      };

      const response = await fetch(`/api/workflows/${workflow.id}/schedule`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (response.ok) {
        setShowCreateForm(false);
        resetForm();
        loadSchedules();
      } else {
        const error = await response.json();
        setFormErrors([error.detail || 'Failed to create schedule']);
      }
    } catch (error) {
      console.error('Failed to create schedule:', error);
      setFormErrors(['Failed to create schedule']);
    }
  };

  const handleUpdateSchedule = async () => {
    if (!editingSchedule || !validateForm()) return;

    try {
      const payload = {
        schedule: scheduleForm.scheduleType === 'cron' 
          ? scheduleForm.schedule 
          : scheduleForm.humanPattern,
        config: scheduleForm.config
      };

      const response = await fetch(`/api/workflows/schedules/${editingSchedule.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (response.ok) {
        setEditingSchedule(null);
        resetForm();
        loadSchedules();
      } else {
        const error = await response.json();
        setFormErrors([error.detail || 'Failed to update schedule']);
      }
    } catch (error) {
      console.error('Failed to update schedule:', error);
      setFormErrors(['Failed to update schedule']);
    }
  };

  const handleDeleteSchedule = async (scheduleId: string) => {
    if (!window.confirm('Are you sure you want to delete this schedule?')) return;

    try {
      const response = await fetch(`/api/workflows/schedules/${scheduleId}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        loadSchedules();
      }
    } catch (error) {
      console.error('Failed to delete schedule:', error);
    }
  };

  const handleToggleSchedule = async (schedule: WorkflowTrigger) => {
    try {
      const action = schedule.is_active ? 'pause' : 'resume';
      const response = await fetch(`/api/workflows/schedules/${schedule.id}/${action}`, {
        method: 'POST'
      });

      if (response.ok) {
        loadSchedules();
      }
    } catch (error) {
      console.error('Failed to toggle schedule:', error);
    }
  };

  const resetForm = () => {
    setScheduleForm({
      name: '',
      schedule: '',
      scheduleType: 'human',
      humanPattern: 'daily',
      timezone: 'UTC',
      startDate: '',
      endDate: '',
      maxExecutions: '',
      config: {}
    });
    setFormErrors([]);
  };

  const startEditing = (schedule: WorkflowTrigger) => {
    setEditingSchedule(schedule);
    setScheduleForm({
      name: schedule.name,
      schedule: schedule.schedule,
      scheduleType: 'cron', // Assume cron for editing
      humanPattern: 'daily',
      timezone: 'UTC',
      startDate: '',
      endDate: '',
      maxExecutions: '',
      config: schedule.config || {}
    });
    setShowCreateForm(true);
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
            <h1 className="text-2xl font-bold text-gray-900">Workflow Scheduler</h1>
            <p className="text-gray-600">{workflow.name}</p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <button
            onClick={() => {
              resetForm();
              setShowCreateForm(true);
            }}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
          >
            <Plus className="h-4 w-4" />
            New Schedule
          </button>
        </div>
      </div>
    </div>
  );

  const renderScheduleForm = () => {
    if (!showCreateForm) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
          <div className="p-6 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-bold text-gray-900">
                {editingSchedule ? 'Edit Schedule' : 'Create Schedule'}
              </h2>
              <button
                onClick={() => {
                  setShowCreateForm(false);
                  setEditingSchedule(null);
                  resetForm();
                }}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="h-6 w-6" />
              </button>
            </div>
          </div>
          
          <div className="p-6 space-y-6">
            {formErrors.length > 0 && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex">
                  <AlertCircle className="h-5 w-5 text-red-400" />
                  <div className="ml-3">
                    <h3 className="text-sm font-medium text-red-800">Validation Errors</h3>
                    <ul className="mt-2 text-sm text-red-700 list-disc pl-5 space-y-1">
                      {formErrors.map((error, index) => (
                        <li key={index}>{error}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Schedule Name
              </label>
              <input
                type="text"
                value={scheduleForm.name}
                onChange={(e) => setScheduleForm(prev => ({ ...prev, name: e.target.value }))}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="e.g., Daily report generation"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Schedule Type
              </label>
              <div className="flex gap-4">
                <label className="flex items-center">
                  <input
                    type="radio"
                    value="human"
                    checked={scheduleForm.scheduleType === 'human'}
                    onChange={(e) => setScheduleForm(prev => ({ ...prev, scheduleType: e.target.value as any }))}
                    className="mr-2"
                  />
                  Human-readable
                </label>
                <label className="flex items-center">
                  <input
                    type="radio"
                    value="cron"
                    checked={scheduleForm.scheduleType === 'cron'}
                    onChange={(e) => setScheduleForm(prev => ({ ...prev, scheduleType: e.target.value as any }))}
                    className="mr-2"
                  />
                  Cron expression
                </label>
              </div>
            </div>

            {scheduleForm.scheduleType === 'human' ? (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Schedule Pattern
                </label>
                <select
                  value={scheduleForm.humanPattern}
                  onChange={(e) => setScheduleForm(prev => ({ ...prev, humanPattern: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  {HUMAN_PATTERNS.map((pattern) => (
                    <option key={pattern.value} value={pattern.value}>
                      {pattern.label}
                    </option>
                  ))}
                </select>
              </div>
            ) : (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Cron Expression
                </label>
                <input
                  type="text"
                  value={scheduleForm.schedule}
                  onChange={(e) => setScheduleForm(prev => ({ ...prev, schedule: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 font-mono"
                  placeholder="0 9 * * 1-5"
                />
                <p className="text-sm text-gray-500 mt-1">
                  Format: minute hour day month weekday (e.g., "0 9 * * 1-5" for weekdays at 9 AM)
                </p>
              </div>
            )}

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Timezone
                </label>
                <select
                  value={scheduleForm.timezone}
                  onChange={(e) => setScheduleForm(prev => ({ ...prev, timezone: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  {TIMEZONES.map((tz) => (
                    <option key={tz} value={tz}>{tz}</option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Max Executions (optional)
                </label>
                <input
                  type="number"
                  value={scheduleForm.maxExecutions}
                  onChange={(e) => setScheduleForm(prev => ({ ...prev, maxExecutions: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  placeholder="Unlimited"
                  min="1"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Start Date (optional)
                </label>
                <input
                  type="datetime-local"
                  value={scheduleForm.startDate}
                  onChange={(e) => setScheduleForm(prev => ({ ...prev, startDate: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  End Date (optional)
                </label>
                <input
                  type="datetime-local"
                  value={scheduleForm.endDate}
                  onChange={(e) => setScheduleForm(prev => ({ ...prev, endDate: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>
            </div>
          </div>
          
          <div className="px-6 py-4 border-t border-gray-200 flex justify-end gap-3">
            <button
              onClick={() => {
                setShowCreateForm(false);
                setEditingSchedule(null);
                resetForm();
              }}
              className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              onClick={editingSchedule ? handleUpdateSchedule : handleCreateSchedule}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
            >
              <Save className="h-4 w-4" />
              {editingSchedule ? 'Update Schedule' : 'Create Schedule'}
            </button>
          </div>
        </div>
      </div>
    );
  };

  const renderSchedulesList = () => (
    <div className="bg-white shadow overflow-hidden">
      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-2 text-gray-600">Loading schedules...</span>
        </div>
      ) : schedules.length === 0 ? (
        <div className="text-center py-12">
          <Clock className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No schedules found</h3>
          <p className="text-gray-600 mb-4">Create your first schedule to automate this workflow.</p>
          <button
            onClick={() => {
              resetForm();
              setShowCreateForm(true);
            }}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Create Schedule
          </button>
        </div>
      ) : (
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Schedule
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Pattern
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Status
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Last Run
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Next Run
              </th>
              <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {schedules.map((schedule) => (
              <tr key={schedule.id} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap">
                  <div>
                    <div className="text-sm font-medium text-gray-900">{schedule.name}</div>
                    <div className="text-sm text-gray-500">
                      {schedule.execution_count} executions
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <code className="text-sm bg-gray-100 px-2 py-1 rounded">
                    {schedule.schedule}
                  </code>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center gap-2">
                    {schedule.is_active ? (
                      <CheckCircle className="h-5 w-5 text-green-500" />
                    ) : (
                      <Pause className="h-5 w-5 text-gray-400" />
                    )}
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      schedule.is_active 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-gray-100 text-gray-800'
                    }`}>
                      {schedule.is_active ? 'Active' : 'Paused'}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {schedule.last_executed_at 
                    ? new Date(schedule.last_executed_at).toLocaleString()
                    : 'Never'
                  }
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  {schedule.next_execution_at 
                    ? new Date(schedule.next_execution_at).toLocaleString()
                    : '-'
                  }
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                  <div className="flex items-center justify-end gap-2">
                    <button
                      onClick={() => handleToggleSchedule(schedule)}
                      className={`p-1 rounded ${
                        schedule.is_active 
                          ? 'text-yellow-600 hover:bg-yellow-100' 
                          : 'text-green-600 hover:bg-green-100'
                      }`}
                      title={schedule.is_active ? 'Pause schedule' : 'Resume schedule'}
                    >
                      {schedule.is_active ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                    </button>
                    
                    <button
                      onClick={() => startEditing(schedule)}
                      className="p-1 text-blue-600 hover:bg-blue-100 rounded"
                      title="Edit schedule"
                    >
                      <Edit className="h-4 w-4" />
                    </button>
                    
                    <button
                      onClick={() => handleDeleteSchedule(schedule.id)}
                      className="p-1 text-red-600 hover:bg-red-100 rounded"
                      title="Delete schedule"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );

  return (
    <div className="h-screen flex flex-col">
      {renderHeader()}
      
      <div className="flex-1 overflow-y-auto p-6">
        {renderSchedulesList()}
      </div>
      
      {renderScheduleForm()}
    </div>
  );
};