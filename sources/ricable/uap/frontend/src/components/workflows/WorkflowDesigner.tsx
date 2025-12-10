// frontend/src/components/workflows/WorkflowDesigner.tsx
// Agent 25: Advanced Workflow Automation - Visual Designer with Drag-and-Drop

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';

// Types for workflow system
interface WorkflowNode {
  id: string;
  type: 'trigger' | 'action' | 'condition' | 'data' | 'ai_agent';
  name: string;
  config: Record<string, any>;
  position: { x: number; y: number };
  connections: string[];
}

interface WorkflowEdge {
  id: string;
  source: string;
  target: string;
  label?: string;
  condition?: string;
}

interface Workflow {
  id: string;
  name: string;
  description: string;
  nodes: WorkflowNode[];
  edges: WorkflowEdge[];
  status: 'draft' | 'active' | 'paused' | 'error';
  created_at: string;
  updated_at: string;
  triggers: string[];
  variables: Record<string, any>;
}

// Node templates
const NODE_TEMPLATES = {
  triggers: [
    {
      id: 'webhook',
      name: 'Webhook Trigger',
      icon: '🔗',
      description: 'Triggered by HTTP webhook',
      config: { url: '', method: 'POST', headers: {} }
    },
    {
      id: 'schedule',
      name: 'Schedule Trigger',
      icon: '⏰',
      description: 'Run on schedule (cron)',
      config: { cron: '0 9 * * 1-5', timezone: 'UTC' }
    },
    {
      id: 'file_upload',
      name: 'File Upload',
      icon: '📁',
      description: 'Triggered by file upload',
      config: { extensions: ['.pdf', '.docx'], max_size: '10MB' }
    }
  ],
  actions: [
    {
      id: 'send_email',
      name: 'Send Email',
      icon: '📧',
      description: 'Send email notification',
      config: { to: '', subject: '', template: '' }
    },
    {
      id: 'call_api',
      name: 'API Call',
      icon: '🌐',
      description: 'Make HTTP API call',
      config: { url: '', method: 'GET', headers: {}, body: {} }
    },
    {
      id: 'transform_data',
      name: 'Transform Data',
      icon: '🔄',
      description: 'Transform data format',
      config: { mapping: {}, filters: [] }
    }
  ],
  ai_agents: [
    {
      id: 'copilot_agent',
      name: 'CopilotKit Agent',
      icon: '🤖',
      description: 'AI code assistance',
      config: { model: 'gpt-4', prompt_template: '', max_tokens: 1000 }
    },
    {
      id: 'agno_agent',
      name: 'Agno Document Processor',
      icon: '📄',
      description: 'Document analysis with Agno',
      config: { analysis_type: 'full', output_format: 'json' }
    },
    {
      id: 'mastra_agent',
      name: 'Mastra Workflow',
      icon: '⚡',
      description: 'Execute Mastra workflow',
      config: { workflow_id: '', parameters: {} }
    }
  ],
  conditions: [
    {
      id: 'if_condition',
      name: 'If/Else',
      icon: '❓',
      description: 'Conditional branch',
      config: { condition: '', true_path: '', false_path: '' }
    },
    {
      id: 'filter',
      name: 'Filter',
      icon: '🔍',
      description: 'Filter data based on criteria',
      config: { field: '', operator: 'equals', value: '' }
    }
  ]
};

// Workflow Node Component
interface WorkflowNodeComponentProps {
  node: WorkflowNode;
  onSelect: (nodeId: string) => void;
  onDelete: (nodeId: string) => void;
  onEdit: (nodeId: string) => void;
  isSelected: boolean;
}

const WorkflowNodeComponent: React.FC<WorkflowNodeComponentProps> = ({
  node,
  onSelect,
  onDelete,
  onEdit,
  isSelected
}) => {
  const getNodeColor = (type: string) => {
    switch (type) {
      case 'trigger': return 'bg-green-100 border-green-300';
      case 'action': return 'bg-blue-100 border-blue-300';
      case 'condition': return 'bg-yellow-100 border-yellow-300';
      case 'ai_agent': return 'bg-purple-100 border-purple-300';
      default: return 'bg-gray-100 border-gray-300';
    }
  };

  return (
    <div
      className={`
        absolute p-3 rounded-lg border-2 cursor-pointer min-w-32 max-w-48
        ${getNodeColor(node.type)}
        ${isSelected ? 'ring-2 ring-blue-500' : ''}
        hover:shadow-md transition-all duration-200
      `}
      style={{
        left: node.position.x,
        top: node.position.y,
        transform: 'translate(-50%, -50%)'
      }}
      onClick={() => onSelect(node.id)}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-lg">{NODE_TEMPLATES.triggers.find(t => t.id === node.type)?.icon || '⚙️'}</span>
        <div className="flex space-x-1">
          <button
            onClick={(e) => {
              e.stopPropagation();
              onEdit(node.id);
            }}
            className="text-xs text-gray-500 hover:text-blue-500"
          >
            ✏️
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete(node.id);
            }}
            className="text-xs text-gray-500 hover:text-red-500"
          >
            🗑️
          </button>
        </div>
      </div>
      <div className="text-sm font-medium text-gray-800">{node.name}</div>
      <div className="text-xs text-gray-600 mt-1">
        {Object.keys(node.config).length} config items
      </div>
    </div>
  );
};

// Node Palette Component
interface NodePaletteProps {
  onAddNode: (template: any, category: string) => void;
}

const NodePalette: React.FC<NodePaletteProps> = ({ onAddNode }) => {
  const [activeCategory, setActiveCategory] = useState('triggers');

  return (
    <div className="w-64 bg-white border-r border-gray-200 h-full overflow-y-auto">
      <div className="p-4 border-b border-gray-200">
        <h3 className="text-lg font-semibold text-gray-800">Workflow Components</h3>
      </div>
      
      {/* Category Tabs */}
      <div className="flex flex-wrap p-2 border-b border-gray-200">
        {Object.keys(NODE_TEMPLATES).map((category) => (
          <button
            key={category}
            onClick={() => setActiveCategory(category)}
            className={`
              px-3 py-1 text-sm rounded-md mr-2 mb-2 capitalize
              ${activeCategory === category 
                ? 'bg-blue-500 text-white' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }
            `}
          >
            {category}
          </button>
        ))}
      </div>

      {/* Node Templates */}
      <div className="p-2">
        {NODE_TEMPLATES[activeCategory as keyof typeof NODE_TEMPLATES]?.map((template) => (
          <div
            key={template.id}
            className="p-3 mb-2 border border-gray-200 rounded-lg cursor-pointer hover:bg-gray-50 hover:border-blue-300 transition-colors"
            onClick={() => onAddNode(template, activeCategory)}
          >
            <div className="flex items-center mb-2">
              <span className="text-lg mr-2">{template.icon}</span>
              <span className="font-medium text-sm">{template.name}</span>
            </div>
            <p className="text-xs text-gray-600">{template.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

// Node Configuration Panel
interface NodeConfigPanelProps {
  node: WorkflowNode | null;
  onUpdateNode: (nodeId: string, updates: Partial<WorkflowNode>) => void;
  onClose: () => void;
}

const NodeConfigPanel: React.FC<NodeConfigPanelProps> = ({ node, onUpdateNode, onClose }) => {
  const [config, setConfig] = useState(node?.config || {});
  const [name, setName] = useState(node?.name || '');

  useEffect(() => {
    if (node) {
      setConfig(node.config);
      setName(node.name);
    }
  }, [node]);

  const handleSave = () => {
    if (node) {
      onUpdateNode(node.id, { config, name });
      onClose();
    }
  };

  if (!node) return null;

  return (
    <div className="w-80 bg-white border-l border-gray-200 h-full overflow-y-auto">
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-800">Configure Node</h3>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            ✕
          </button>
        </div>
      </div>

      <div className="p-4 space-y-4">
        {/* Node Name */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Node Name
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        {/* Configuration Fields */}
        {Object.entries(config).map(([key, value]) => (
          <div key={key}>
            <label className="block text-sm font-medium text-gray-700 mb-1 capitalize">
              {key.replace(/_/g, ' ')}
            </label>
            <input
              type="text"
              value={typeof value === 'object' ? JSON.stringify(value) : value}
              onChange={(e) => {
                let newValue: any = e.target.value;
                try {
                  // Try to parse as JSON for objects
                  if (typeof value === 'object') {
                    newValue = JSON.parse(e.target.value);
                  }
                } catch {
                  // Keep as string if JSON parsing fails
                }
                setConfig(prev => ({ ...prev, [key]: newValue }));
              }}
              className="w-full p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
        ))}

        {/* Save Button */}
        <div className="pt-4 border-t border-gray-200">
          <button
            onClick={handleSave}
            className="w-full bg-blue-500 text-white py-2 px-4 rounded-md hover:bg-blue-600 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
          >
            Save Configuration
          </button>
        </div>
      </div>
    </div>
  );
};

// Main Workflow Designer Component
export const WorkflowDesigner: React.FC = () => {
  const [workflow, setWorkflow] = useState<Workflow>({
    id: 'new-workflow',
    name: 'New Workflow',
    description: 'Workflow description',
    nodes: [],
    edges: [],
    status: 'draft',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    triggers: [],
    variables: {}
  });

  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [showConfigPanel, setShowConfigPanel] = useState(false);
  const canvasRef = useRef<HTMLDivElement>(null);

  // Add new node to workflow
  const handleAddNode = useCallback((template: any, category: string) => {
    const newNode: WorkflowNode = {
      id: `${template.id}_${Date.now()}`,
      type: category.slice(0, -1) as WorkflowNode['type'], // Remove 's' from category
      name: template.name,
      config: { ...template.config },
      position: { x: 300, y: 200 },
      connections: []
    };

    setWorkflow(prev => ({
      ...prev,
      nodes: [...prev.nodes, newNode],
      updated_at: new Date().toISOString()
    }));
  }, []);

  // Update node
  const handleUpdateNode = useCallback((nodeId: string, updates: Partial<WorkflowNode>) => {
    setWorkflow(prev => ({
      ...prev,
      nodes: prev.nodes.map(node => 
        node.id === nodeId ? { ...node, ...updates } : node
      ),
      updated_at: new Date().toISOString()
    }));
  }, []);

  // Delete node
  const handleDeleteNode = useCallback((nodeId: string) => {
    setWorkflow(prev => ({
      ...prev,
      nodes: prev.nodes.filter(node => node.id !== nodeId),
      edges: prev.edges.filter(edge => edge.source !== nodeId && edge.target !== nodeId),
      updated_at: new Date().toISOString()
    }));
    
    if (selectedNodeId === nodeId) {
      setSelectedNodeId(null);
      setShowConfigPanel(false);
    }
  }, [selectedNodeId]);

  // Select node
  const handleSelectNode = useCallback((nodeId: string) => {
    setSelectedNodeId(nodeId);
  }, []);

  // Edit node
  const handleEditNode = useCallback((nodeId: string) => {
    setSelectedNodeId(nodeId);
    setShowConfigPanel(true);
  }, []);

  // Save workflow
  const handleSaveWorkflow = useCallback(async () => {
    try {
      // In a real implementation, this would save to the backend
      console.log('Saving workflow:', workflow);
      
      // Mock API call
      const response = await fetch('/api/workflows', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(workflow)
      });
      
      if (response.ok) {
        alert('Workflow saved successfully!');
      }
    } catch (error) {
      console.error('Failed to save workflow:', error);
      alert('Failed to save workflow');
    }
  }, [workflow]);

  // Execute workflow
  const handleExecuteWorkflow = useCallback(async () => {
    try {
      // In a real implementation, this would trigger workflow execution
      console.log('Executing workflow:', workflow);
      
      const response = await fetch(`/api/workflows/${workflow.id}/execute`, {
        method: 'POST'
      });
      
      if (response.ok) {
        alert('Workflow execution started!');
      }
    } catch (error) {
      console.error('Failed to execute workflow:', error);
      alert('Failed to execute workflow');
    }
  }, [workflow]);

  const selectedNode = workflow.nodes.find(node => node.id === selectedNodeId);

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Node Palette */}
      <NodePalette onAddNode={handleAddNode} />

      {/* Main Canvas */}
      <div className="flex-1 flex flex-col">
        {/* Toolbar */}
        <div className="bg-white border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-xl font-semibold text-gray-800">{workflow.name}</h1>
              <p className="text-sm text-gray-600">{workflow.description}</p>
            </div>
            <div className="flex space-x-2">
              <button
                onClick={handleSaveWorkflow}
                className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
              >
                Save Workflow
              </button>
              <button
                onClick={handleExecuteWorkflow}
                className="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 focus:ring-2 focus:ring-green-500 focus:ring-offset-2"
              >
                Execute
              </button>
            </div>
          </div>
        </div>

        {/* Canvas */}
        <div className="flex-1 relative overflow-hidden">
          <div
            ref={canvasRef}
            className="absolute inset-0 bg-white bg-grid-pattern"
            style={{
              backgroundImage: 'radial-gradient(circle, #e5e7eb 1px, transparent 1px)',
              backgroundSize: '20px 20px'
            }}
          >
            {/* Render Nodes */}
            {workflow.nodes.map((node) => (
              <WorkflowNodeComponent
                key={node.id}
                node={node}
                onSelect={handleSelectNode}
                onDelete={handleDeleteNode}
                onEdit={handleEditNode}
                isSelected={selectedNodeId === node.id}
              />
            ))}

            {/* Render Edges (connections) */}
            <svg className="absolute inset-0 pointer-events-none">
              {workflow.edges.map((edge) => {
                const sourceNode = workflow.nodes.find(n => n.id === edge.source);
                const targetNode = workflow.nodes.find(n => n.id === edge.target);
                
                if (!sourceNode || !targetNode) return null;
                
                return (
                  <line
                    key={edge.id}
                    x1={sourceNode.position.x}
                    y1={sourceNode.position.y}
                    x2={targetNode.position.x}
                    y2={targetNode.position.y}
                    stroke="#6b7280"
                    strokeWidth="2"
                    markerEnd="url(#arrowhead)"
                  />
                );
              })}
              
              {/* Arrow marker definition */}
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

            {/* Empty State */}
            {workflow.nodes.length === 0 && (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center text-gray-500">
                  <div className="text-4xl mb-4">⚡</div>
                  <h3 className="text-lg font-medium mb-2">Start Building Your Workflow</h3>
                  <p className="text-sm">Drag components from the left panel to begin</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Configuration Panel */}
      {showConfigPanel && (
        <NodeConfigPanel
          node={selectedNode || null}
          onUpdateNode={handleUpdateNode}
          onClose={() => setShowConfigPanel(false)}
        />
      )}
    </div>
  );
};

export default WorkflowDesigner;