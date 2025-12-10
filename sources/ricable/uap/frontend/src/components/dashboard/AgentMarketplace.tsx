// Agent marketplace component for discovering and managing agents
import React, { useState, useEffect } from 'react';
import { Search, Filter, Star, Clock, Activity, Zap, MessageSquare, Settings, Play } from 'lucide-react';
import { AgentChat } from '../agents/AgentChat';

interface Agent {
  id: string;
  name: string;
  description: string;
  framework: string;
  status: 'active' | 'inactive' | 'error';
  capabilities: string[];
  rating: number;
  usage_count: number;
  avg_response_time: number;
  last_active: string;
  created_at: string;
  category: string;
  tags: string[];
  version: string;
  cost_per_request?: number;
}

interface AgentStats {
  [agentId: string]: {
    agent_id: string;
    framework: string;
    total_requests: number;
    avg_response_time_ms: number;
    p95_response_time_ms: number;
    p99_response_time_ms: number;
    success_rate: number;
    last_request_time: string | null;
  };
}

const FRAMEWORK_COLORS = {
  copilot: 'bg-blue-100 text-blue-800',
  agno: 'bg-green-100 text-green-800',
  mastra: 'bg-purple-100 text-purple-800',
};

const STATUS_COLORS = {
  active: 'bg-green-100 text-green-800',
  inactive: 'bg-gray-100 text-gray-800',
  error: 'bg-red-100 text-red-800',
};

export const AgentMarketplace: React.FC = () => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [agentStats, setAgentStats] = useState<AgentStats>({});
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedFramework, setSelectedFramework] = useState('all');
  const [sortBy, setSortBy] = useState<'name' | 'rating' | 'usage' | 'response_time'>('rating');
  const [isLoading, setIsLoading] = useState(true);
  const [activeAgent, setActiveAgent] = useState<Agent | null>(null);
  const [showChat, setShowChat] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  // Fetch agents from real API
  const fetchAgents = async () => {
    try {
      const response = await fetch('/api/agents');
      if (response.ok) {
        const data = await response.json();
        
        // Transform backend agent data to frontend format
        const agentList = data.agents || [];
        const transformedAgents: Agent[] = agentList.map((agentData: any) => {
          // Use real agent data from the enhanced backend API
          return {
            id: agentData.id,
            name: agentData.name,
            description: agentData.description,
            framework: agentData.framework,
            status: agentData.status,
            capabilities: agentData.capabilities || [],
            rating: agentData.rating || 4.5,
            usage_count: agentData.usage_count || 0,
            avg_response_time: agentData.avg_response_time || 100,
            last_active: agentData.last_active || 'recently',
            created_at: agentData.created_at || '2024-01-01',
            category: agentData.category || 'General',
            tags: agentData.tags || [],
            version: agentData.version || '1.0.0',
            cost_per_request: agentData.cost_per_request || 0.001,
          } as Agent;
        });
        
        setAgents(transformedAgents);
      } else {
        console.error('Failed to fetch agents:', response.statusText);
        setAgents([]);
      }
    } catch (error) {
      console.error('Failed to fetch agents:', error);
      setAgents([]);
    }
  };

  const fetchAgentStats = async () => {
    try {
      const response = await fetch('/api/monitoring/agents');
      if (response.ok) {
        const stats = await response.json();
        setAgentStats(stats);
      } else {
        console.error('Failed to fetch agent stats:', response.statusText);
        setAgentStats({});
      }
    } catch (error) {
      console.error('Failed to fetch agent stats:', error);
      setAgentStats({});
    }
  };

  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      await Promise.all([
        fetchAgents(),
        fetchAgentStats()
      ]);
      setIsLoading(false);
    };

    loadData();
    
    // Update stats every 30 seconds
    const interval = setInterval(fetchAgentStats, 30000);
    return () => clearInterval(interval);
  }, []);

  const categories = ['all', ...Array.from(new Set(agents.map(agent => agent.category)))];
  const frameworks = ['all', 'copilot', 'agno', 'mastra'];

  const filteredAgents = agents
    .filter(agent => {
      const matchesSearch = agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           agent.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           agent.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
      const matchesCategory = selectedCategory === 'all' || agent.category === selectedCategory;
      const matchesFramework = selectedFramework === 'all' || agent.framework === selectedFramework;
      
      return matchesSearch && matchesCategory && matchesFramework;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.name.localeCompare(b.name);
        case 'rating':
          return b.rating - a.rating;
        case 'usage':
          return b.usage_count - a.usage_count;
        case 'response_time':
          return a.avg_response_time - b.avg_response_time;
        default:
          return 0;
      }
    });

  const handleAgentInteract = (agent: Agent) => {
    // Navigate to agent chat interface by setting the active agent
    setActiveAgent(agent);
    setShowChat(true);
  };

  const handleAgentConfigure = (agent: Agent) => {
    // Navigate to agent configuration
    setActiveAgent(agent);
    setShowSettings(true);
  };

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="bg-gray-200 rounded-lg h-64"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <h1 className="text-3xl font-bold text-gray-900">Agent Marketplace</h1>
        <div className="text-sm text-gray-500">
          {filteredAgents.length} agents available
        </div>
      </div>

      {/* Filters and Search */}
      <div className="bg-white p-4 rounded-lg border border-gray-200">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
            <input
              type="text"
              placeholder="Search agents..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          {/* Category Filter */}
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {categories.map(category => (
              <option key={category} value={category}>
                {category === 'all' ? 'All Categories' : category}
              </option>
            ))}
          </select>

          {/* Framework Filter */}
          <select
            value={selectedFramework}
            onChange={(e) => setSelectedFramework(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {frameworks.map(framework => (
              <option key={framework} value={framework}>
                {framework === 'all' ? 'All Frameworks' : framework.charAt(0).toUpperCase() + framework.slice(1)}
              </option>
            ))}
          </select>

          {/* Sort */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            <option value="rating">Sort by Rating</option>
            <option value="usage">Sort by Usage</option>
            <option value="response_time">Sort by Speed</option>
            <option value="name">Sort by Name</option>
          </select>
        </div>
      </div>

      {/* Agent Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {filteredAgents.map((agent) => {
          const stats = agentStats[agent.id];
          const actualResponseTime = stats?.avg_response_time_ms || agent.avg_response_time;
          const successRate = stats?.success_rate || 0.95; // Default success rate

          return (
            <div key={agent.id} className="bg-white rounded-lg border border-gray-200 hover:shadow-lg transition-shadow">
              {/* Agent Header */}
              <div className="p-6 border-b border-gray-100">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">{agent.name}</h3>
                    <p className="text-sm text-gray-500">{agent.category} • v{agent.version}</p>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${FRAMEWORK_COLORS[agent.framework as keyof typeof FRAMEWORK_COLORS]}`}>
                      {agent.framework}
                    </span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${STATUS_COLORS[agent.status as keyof typeof STATUS_COLORS]}`}>
                      {agent.status}
                    </span>
                  </div>
                </div>

                <p className="text-sm text-gray-600 mb-4">{agent.description}</p>

                {/* Rating and Stats */}
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center">
                    <Star className="h-4 w-4 text-yellow-400 fill-current" />
                    <span className="ml-1 text-sm font-medium">{agent.rating}</span>
                  </div>
                  <div className="text-xs text-gray-500">
                    {agent.usage_count} uses
                  </div>
                </div>

                {/* Performance Metrics */}
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="text-center">
                    <div className="text-lg font-semibold text-gray-900">{Math.round(actualResponseTime)}ms</div>
                    <div className="text-xs text-gray-500">Avg Response</div>
                  </div>
                  <div className="text-center">
                    <div className="text-lg font-semibold text-green-600">{Math.round(successRate * 100)}%</div>
                    <div className="text-xs text-gray-500">Success Rate</div>
                  </div>
                </div>

                {/* Capabilities */}
                <div className="mb-4">
                  <div className="text-xs font-medium text-gray-700 mb-2">Capabilities</div>
                  <div className="flex flex-wrap gap-1">
                    {agent.capabilities.slice(0, 3).map((capability, index) => (
                      <span key={index} className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded">
                        {capability}
                      </span>
                    ))}
                    {agent.capabilities.length > 3 && (
                      <span className="px-2 py-1 bg-gray-100 text-gray-500 text-xs rounded">
                        +{agent.capabilities.length - 3} more
                      </span>
                    )}
                  </div>
                </div>

                {/* Cost */}
                {agent.cost_per_request && (
                  <div className="text-xs text-gray-500 mb-4">
                    ${agent.cost_per_request.toFixed(3)} per request
                  </div>
                )}

                {/* Last Active */}
                <div className="flex items-center text-xs text-gray-500 mb-4">
                  <Clock className="h-3 w-3 mr-1" />
                  Last active: {agent.last_active}
                </div>
              </div>

              {/* Actions */}
              <div className="px-6 py-4 bg-gray-50 flex space-x-2">
                <button
                  onClick={() => handleAgentInteract(agent)}
                  className="flex-1 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center text-sm font-medium"
                >
                  <MessageSquare className="h-4 w-4 mr-2" />
                  Chat
                </button>
                <button
                  onClick={() => handleAgentConfigure(agent)}
                  className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-100 transition-colors flex items-center justify-center"
                >
                  <Settings className="h-4 w-4" />
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Empty State */}
      {filteredAgents.length === 0 && (
        <div className="text-center py-12">
          <Filter className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No agents found</h3>
          <p className="text-gray-500">Try adjusting your search or filter criteria</p>
        </div>
      )}

      {/* Chat Interface */}
      {showChat && activeAgent && (
        <AgentChat
          agentId={activeAgent.id}
          agentName={activeAgent.name}
          framework={activeAgent.framework}
          onClose={() => setShowChat(false)}
        />
      )}

      {/* Settings Modal */}
      {showSettings && activeAgent && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-lg">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">Settings for {activeAgent.name}</h2>
              <button
                onClick={() => setShowSettings(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Agent Configuration
                </label>
                <textarea
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  rows={4}
                  placeholder="Configuration settings would be available here..."
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Enable Agent
                </label>
                <input type="checkbox" className="rounded" defaultChecked />
              </div>
              <div className="flex justify-end space-x-2">
                <button
                  onClick={() => setShowSettings(false)}
                  className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-100"
                >
                  Cancel
                </button>
                <button className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700">
                  Save
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};