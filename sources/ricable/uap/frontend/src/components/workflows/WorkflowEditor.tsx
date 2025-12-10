/**
 * Workflow Editor Component
 * 
 * Visual drag-and-drop workflow editor with step configuration,
 * flow connections, and real-time validation.
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import {
  Save,
  ArrowLeft,
  Play,
  Plus,
  Settings,
  Trash2,
  Copy,
  Eye,
  Code,
  Grid,
  Zap,
  MessageSquare,
  Database,
  Mail,
  Webhook,
  Clock
} from 'lucide-react';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';

interface WorkflowStep {
  id: string;
  type: 'agent' | 'condition' | 'parallel' | 'transform' | 'delay' | 'webhook' | 'api_call' | 'email' | 'notification';
  name: string;
  config: any;
  position: { x: number; y: number };
  connections: string[];
}

interface Workflow {
  id?: string;
  name: string;
  description: string;
  definition: {
    steps: WorkflowStep[];
    variables: any;
  };
  variables: any;
  status: string;
}

interface WorkflowEditorProps {
  workflow?: Workflow | null;
  onSave: (workflow: any) => void;
  onCancel: () => void;
}

const STEP_TYPES = [
  {
    id: 'agent',
    name: 'Agent Call',
    icon: MessageSquare,
    description: 'Call an AI agent to process data',
    color: 'blue'
  },
  {
    id: 'condition',
    name: 'Condition',
    icon: Zap,
    description: 'Branch workflow based on conditions',
    color: 'yellow'
  },
  {
    id: 'parallel',
    name: 'Parallel',
    icon: Grid,
    description: 'Execute multiple steps in parallel',
    color: 'purple'
  },
  {
    id: 'transform',
    name: 'Transform',
    icon: Database,
    description: 'Transform or manipulate data',
    color: 'green'
  },
  {
    id: 'delay',
    name: 'Delay',
    icon: Clock,
    description: 'Add a time delay to the workflow',
    color: 'orange'
  },
  {
    id: 'webhook',
    name: 'Webhook',
    icon: Webhook,
    description: 'Send HTTP request to external service',
    color: 'indigo'
  },
  {
    id: 'email',
    name: 'Email',
    icon: Mail,
    description: 'Send email notification',
    color: 'red'
  }
];

export const WorkflowEditor: React.FC<WorkflowEditorProps> = ({
  workflow,
  onSave,
  onCancel
}) => {
  const [workflowData, setWorkflowData] = useState<Workflow>({
    name: workflow?.name || '',
    description: workflow?.description || '',
    definition: workflow?.definition || { steps: [], variables: {} },
    variables: workflow?.variables || {},
    status: workflow?.status || 'draft'
  });
  
  const [selectedStep, setSelectedStep] = useState<WorkflowStep | null>(null);
  const [viewMode, setViewMode] = useState<'visual' | 'code'>('visual');
  const [isDirty, setIsDirty] = useState(false);
  const [errors, setErrors] = useState<string[]>([]);
  const [draggedStepType, setDraggedStepType] = useState<string | null>(null);
  
  const canvasRef = useRef<HTMLDivElement>(null);
  const [canvasOffset, setCanvasOffset] = useState({ x: 0, y: 0 });
  const [scale, setScale] = useState(1);

  useEffect(() => {
    setIsDirty(true);
  }, [workflowData]);

  const generateStepId = () => {
    return `step_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  };

  const addStep = useCallback((stepType: string, position?: { x: number; y: number }) => {
    const stepTypeInfo = STEP_TYPES.find(t => t.id === stepType);
    if (!stepTypeInfo) return;

    const newStep: WorkflowStep = {
      id: generateStepId(),
      type: stepType as any,
      name: `${stepTypeInfo.name} ${workflowData.definition.steps.length + 1}`,
      config: getDefaultConfig(stepType),
      position: position || { x: 100, y: 100 + (workflowData.definition.steps.length * 120) },
      connections: []
    };

    setWorkflowData(prev => ({
      ...prev,
      definition: {
        ...prev.definition,
        steps: [...prev.definition.steps, newStep]
      }
    }));

    setSelectedStep(newStep);
  }, [workflowData.definition.steps.length]);

  const getDefaultConfig = (stepType: string) => {
    switch (stepType) {
      case 'agent':
        return {
          agent_id: 'auto-agent',
          framework: 'auto',
          message: 'Process: {input_data}',
          output_mapping: {}
        };
      case 'condition':
        return {
          condition: 'input_value > 0',
          true_action: {},
          false_action: {}
        };
      case 'parallel':
        return {
          steps: []
        };
      case 'transform':
        return {
          transformations: {
            output: 'input_data'
          }
        };
      case 'delay':
        return {
          delay_seconds: 5,
          delay_type: 'fixed'
        };
      case 'webhook':
        return {
          url: 'https://api.example.com/webhook',
          method: 'POST',
          headers: {},
          payload: {}
        };
      case 'email':
        return {
          to: ['user@example.com'],
          subject: 'Workflow Notification',
          body: 'Your workflow has completed.'
        };
      default:
        return {};
    }
  };

  const updateStep = useCallback((stepId: string, updates: Partial<WorkflowStep>) => {
    setWorkflowData(prev => ({
      ...prev,
      definition: {
        ...prev.definition,
        steps: prev.definition.steps.map(step =>
          step.id === stepId ? { ...step, ...updates } : step
        )
      }
    }));

    if (selectedStep?.id === stepId) {
      setSelectedStep(prev => prev ? { ...prev, ...updates } : null);
    }
  }, [selectedStep]);

  const deleteStep = useCallback((stepId: string) => {
    setWorkflowData(prev => ({
      ...prev,
      definition: {
        ...prev.definition,
        steps: prev.definition.steps.filter(step => step.id !== stepId)
      }
    }));

    if (selectedStep?.id === stepId) {
      setSelectedStep(null);
    }
  }, [selectedStep]);

  const connectSteps = useCallback((fromStepId: string, toStepId: string) => {
    setWorkflowData(prev => ({
      ...prev,
      definition: {
        ...prev.definition,
        steps: prev.definition.steps.map(step =>
          step.id === fromStepId
            ? { ...step, connections: [...step.connections, toStepId] }
            : step
        )
      }
    }));
  }, []);

  const validateWorkflow = () => {
    const errors: string[] = [];
    
    if (!workflowData.name.trim()) {
      errors.push('Workflow name is required');
    }
    
    if (workflowData.definition.steps.length === 0) {
      errors.push('At least one step is required');
    }
    
    // Validate each step
    workflowData.definition.steps.forEach((step, index) => {
      if (!step.name.trim()) {
        errors.push(`Step ${index + 1}: Name is required`);
      }
      
      // Type-specific validation
      switch (step.type) {
        case 'agent':
          if (!step.config.agent_id) {
            errors.push(`Step ${index + 1}: Agent ID is required`);
          }
          if (!step.config.message) {
            errors.push(`Step ${index + 1}: Message template is required`);
          }
          break;
        case 'condition':
          if (!step.config.condition) {
            errors.push(`Step ${index + 1}: Condition expression is required`);
          }
          break;
        case 'webhook':
          if (!step.config.url) {
            errors.push(`Step ${index + 1}: Webhook URL is required`);
          }
          break;
        case 'email':
          if (!step.config.to || step.config.to.length === 0) {
            errors.push(`Step ${index + 1}: Email recipients are required`);
          }
          break;
      }
    });
    
    setErrors(errors);
    return errors.length === 0;
  };

  const handleSave = () => {
    if (!validateWorkflow()) {
      return;
    }

    const workflowToSave = {
      ...workflowData,
      definition: {
        ...workflowData.definition,
        steps: workflowData.definition.steps.map(step => ({
          id: step.id,
          type: step.type,
          name: step.name,
          config: step.config,
          order_index: workflowData.definition.steps.indexOf(step)
        }))
      }
    };

    onSave(workflowToSave);
  };

  const handleCanvasDrop = (e: React.DragEvent) => {
    e.preventDefault();
    
    if (!draggedStepType || !canvasRef.current) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const position = {
      x: (e.clientX - rect.left - canvasOffset.x) / scale,
      y: (e.clientY - rect.top - canvasOffset.y) / scale
    };
    
    addStep(draggedStepType, position);
    setDraggedStepType(null);
  };

  const renderStepPalette = () => (
    <div className="w-64 bg-gray-50 border-r border-gray-200 p-4">
      <h3 className="text-sm font-medium text-gray-900 mb-4">Step Types</h3>
      <div className="space-y-2">
        {STEP_TYPES.map((stepType) => {
          const Icon = stepType.icon;
          return (
            <div
              key={stepType.id}
              draggable
              onDragStart={() => setDraggedStepType(stepType.id)}
              onDragEnd={() => setDraggedStepType(null)}
              className="p-3 bg-white rounded-lg border border-gray-200 cursor-grab hover:shadow-md transition-shadow"
            >
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg bg-${stepType.color}-100`}>
                  <Icon className={`h-4 w-4 text-${stepType.color}-600`} />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium text-gray-900">{stepType.name}</p>
                  <p className="text-xs text-gray-500">{stepType.description}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );

  const renderWorkflowCanvas = () => (
    <div className="flex-1 relative overflow-hidden">
      <div
        ref={canvasRef}
        className="w-full h-full bg-white"
        onDrop={handleCanvasDrop}
        onDragOver={(e) => e.preventDefault()}
        style={{
          backgroundImage: 'radial-gradient(circle, #e5e7eb 1px, transparent 1px)',
          backgroundSize: '20px 20px'
        }}
      >
        {workflowData.definition.steps.map((step) => (
          <WorkflowStepNode
            key={step.id}
            step={step}
            isSelected={selectedStep?.id === step.id}
            onSelect={() => setSelectedStep(step)}
            onUpdate={(updates) => updateStep(step.id, updates)}
            onDelete={() => deleteStep(step.id)}
            scale={scale}
          />
        ))}
        
        {/* Connection lines */}
        <svg className="absolute inset-0 pointer-events-none">
          {workflowData.definition.steps.map((step) =>
            step.connections.map((connectionId) => {
              const targetStep = workflowData.definition.steps.find(s => s.id === connectionId);
              if (!targetStep) return null;
              
              return (
                <line
                  key={`${step.id}-${connectionId}`}
                  x1={step.position.x + 100}
                  y1={step.position.y + 30}
                  x2={targetStep.position.x}
                  y2={targetStep.position.y + 30}
                  stroke="#6b7280"
                  strokeWidth="2"
                  markerEnd="url(#arrowhead)"
                />
              );
            })
          )}
          
          <defs>
            <marker
              id="arrowhead"
              markerWidth="10"
              markerHeight="7"
              refX="9"
              refY="3.5"
              orient="auto"
            >
              <polygon
                points="0 0, 10 3.5, 0 7"
                fill="#6b7280"
              />
            </marker>
          </defs>
        </svg>
      </div>
    </div>
  );

  const renderStepConfig = () => {
    if (!selectedStep) {
      return (
        <div className="w-80 bg-gray-50 border-l border-gray-200 p-4">
          <div className="text-center text-gray-500 mt-20">
            <Settings className="h-12 w-12 mx-auto mb-4 text-gray-300" />
            <p>Select a step to configure</p>
          </div>
        </div>
      );
    }

    return (
      <div className="w-80 bg-gray-50 border-l border-gray-200 p-4 overflow-y-auto">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-gray-900">Step Configuration</h3>
          <button
            onClick={() => deleteStep(selectedStep.id)}
            className="p-1 text-red-600 hover:bg-red-100 rounded"
          >
            <Trash2 className="h-4 w-4" />
          </button>
        </div>
        
        <StepConfigForm
          step={selectedStep}
          onChange={(updates) => updateStep(selectedStep.id, updates)}
        />
      </div>
    );
  };

  const renderCodeView = () => (
    <div className="flex-1 p-4">
      <textarea
        value={JSON.stringify(workflowData.definition, null, 2)}
        onChange={(e) => {
          try {
            const parsed = JSON.parse(e.target.value);
            setWorkflowData(prev => ({ ...prev, definition: parsed }));
          } catch (error) {
            // Invalid JSON, don't update
          }
        }}
        className="w-full h-full font-mono text-sm border border-gray-300 rounded-lg p-4 resize-none"
        placeholder="Workflow definition (JSON)"
      />
    </div>
  );

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={onCancel}
              className="p-2 hover:bg-gray-100 rounded-lg"
            >
              <ArrowLeft className="h-5 w-5" />
            </button>
            
            <div>
              <input
                type="text"
                value={workflowData.name}
                onChange={(e) => setWorkflowData(prev => ({ ...prev, name: e.target.value }))}
                className="text-xl font-semibold bg-transparent border-none outline-none"
                placeholder="Workflow Name"
              />
              <input
                type="text"
                value={workflowData.description}
                onChange={(e) => setWorkflowData(prev => ({ ...prev, description: e.target.value }))}
                className="block text-sm text-gray-600 bg-transparent border-none outline-none mt-1"
                placeholder="Description"
              />
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <div className="flex bg-gray-100 rounded-lg p-1">
              <button
                onClick={() => setViewMode('visual')}
                className={`px-3 py-1 rounded-md text-sm font-medium ${
                  viewMode === 'visual'
                    ? 'bg-white text-gray-900 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <Eye className="h-4 w-4 mr-1 inline" />
                Visual
              </button>
              <button
                onClick={() => setViewMode('code')}
                className={`px-3 py-1 rounded-md text-sm font-medium ${
                  viewMode === 'code'
                    ? 'bg-white text-gray-900 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                <Code className="h-4 w-4 mr-1 inline" />
                Code
              </button>
            </div>
            
            <button
              onClick={handleSave}
              disabled={!isDirty || errors.length > 0}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <Save className="h-4 w-4" />
              Save Workflow
            </button>
          </div>
        </div>
        
        {errors.length > 0 && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <h4 className="text-sm font-medium text-red-800 mb-2">Validation Errors:</h4>
            <ul className="text-sm text-red-700 space-y-1">
              {errors.map((error, index) => (
                <li key={index}>• {error}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {viewMode === 'visual' ? (
          <>
            {renderStepPalette()}
            {renderWorkflowCanvas()}
            {renderStepConfig()}
          </>
        ) : (
          renderCodeView()
        )}
      </div>
    </div>
  );
};

// Additional components would be defined here
const WorkflowStepNode: React.FC<any> = ({ step, isSelected, onSelect, onUpdate, onDelete, scale }) => {
  const stepType = STEP_TYPES.find(t => t.id === step.type);
  const Icon = stepType?.icon || Settings;
  
  return (
    <div
      className={`absolute bg-white rounded-lg border-2 p-4 cursor-pointer min-w-[200px] ${
        isSelected ? 'border-blue-500 shadow-lg' : 'border-gray-200 hover:border-gray-300'
      }`}
      style={{
        left: step.position.x,
        top: step.position.y,
        transform: `scale(${scale})`
      }}
      onClick={onSelect}
    >
      <div className="flex items-center gap-3">
        <div className={`p-2 rounded-lg bg-${stepType?.color || 'gray'}-100`}>
          <Icon className={`h-4 w-4 text-${stepType?.color || 'gray'}-600`} />
        </div>
        <div className="flex-1 min-w-0">
          <h4 className="font-medium text-gray-900 truncate">{step.name}</h4>
          <p className="text-sm text-gray-500">{stepType?.name}</p>
        </div>
      </div>
    </div>
  );
};

const StepConfigForm: React.FC<any> = ({ step, onChange }) => {
  // This would contain step-specific configuration forms
  return (
    <div className="space-y-4">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Step Name
        </label>
        <input
          type="text"
          value={step.name}
          onChange={(e) => onChange({ name: e.target.value })}
          className="w-full border border-gray-300 rounded-lg px-3 py-2"
        />
      </div>
      
      {/* Step-specific configuration would go here */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Configuration
        </label>
        <textarea
          value={JSON.stringify(step.config, null, 2)}
          onChange={(e) => {
            try {
              const config = JSON.parse(e.target.value);
              onChange({ config });
            } catch (error) {
              // Invalid JSON
            }
          }}
          className="w-full h-40 font-mono text-sm border border-gray-300 rounded-lg p-3"
        />
      </div>
    </div>
  );
};