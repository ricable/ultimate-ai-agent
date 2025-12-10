import { useState, useEffect, useContext, createContext, ReactNode } from 'react';
import { APIClient } from '../services/APIClient';
import { OfflineService } from '../services/OfflineService';
import { useNetwork } from './useNetwork';
import { Logger } from '../utils/Logger';

export interface Agent {
  id: string;
  name: string;
  type: 'copilot' | 'agno' | 'mastra';
  status: 'online' | 'offline' | 'busy';
  description: string;
  lastUsed?: Date;
  capabilities: string[];
  responseTime?: number;
  isActive: boolean;
  version: string;
}

interface AgentsContextType {
  agents: Agent[];
  loading: boolean;
  error: string | null;
  fetchAgents: () => Promise<void>;
  getAgent: (id: string) => Agent | undefined;
  updateAgentStatus: (id: string, status: Agent['status']) => void;
  refreshAgents: () => Promise<void>;
}

const AgentsContext = createContext<AgentsContextType | undefined>(undefined);

export const AgentProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { isOnline } = useNetwork();

  useEffect(() => {
    fetchAgents();
  }, []);

  useEffect(() => {
    if (isOnline) {
      refreshAgents();
    }
  }, [isOnline]);

  const fetchAgents = async () => {
    try {
      setLoading(true);
      setError(null);

      let agentsData: Agent[];

      if (isOnline) {
        // Fetch from API
        agentsData = await APIClient.getAgents();
        // Cache the data
        await OfflineService.cacheAgents(agentsData);
      } else {
        // Load from cache
        agentsData = await OfflineService.getCachedAgents();
      }

      setAgents(agentsData);
      Logger.info('useAgents', `Loaded ${agentsData.length} agents`);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch agents';
      Logger.error('useAgents', 'Failed to fetch agents', err);
      setError(errorMessage);
      
      // Try to load cached data as fallback
      try {
        const cachedAgents = await OfflineService.getCachedAgents();
        setAgents(cachedAgents);
      } catch (cacheError) {
        Logger.error('useAgents', 'Failed to load cached agents', cacheError);
        // Set demo data as last resort
        setAgents(getDemoAgents());
      }
    } finally {
      setLoading(false);
    }
  };

  const refreshAgents = async () => {
    if (!isOnline) return;
    
    try {
      const agentsData = await APIClient.getAgents();
      setAgents(agentsData);
      await OfflineService.cacheAgents(agentsData);
      setError(null);
    } catch (err) {
      Logger.error('useAgents', 'Failed to refresh agents', err);
      // Don't set error on refresh failure, keep existing data
    }
  };

  const getAgent = (id: string): Agent | undefined => {
    return agents.find(agent => agent.id === id);
  };

  const updateAgentStatus = (id: string, status: Agent['status']) => {
    setAgents(prev => 
      prev.map(agent => 
        agent.id === id 
          ? { ...agent, status }
          : agent
      )
    );
  };

  const value: AgentsContextType = {
    agents,
    loading,
    error,
    fetchAgents,
    getAgent,
    updateAgentStatus,
    refreshAgents,
  };

  return (
    <AgentsContext.Provider value={value}>
      {children}
    </AgentsContext.Provider>
  );
};

export const useAgents = (): AgentsContextType => {
  const context = useContext(AgentsContext);
  if (context === undefined) {
    throw new Error('useAgents must be used within an AgentProvider');
  }
  return context;
};

// Demo data for fallback
function getDemoAgents(): Agent[] {
  return [
    {
      id: 'copilot-1',
      name: 'CopilotKit Assistant',
      type: 'copilot',
      status: 'online',
      description: 'AI-powered coding assistant for development tasks and code assistance',
      capabilities: ['Code Generation', 'Debugging', 'Documentation', 'Refactoring'],
      responseTime: 120,
      isActive: true,
      version: '1.0.0',
      lastUsed: new Date(Date.now() - 3600000), // 1 hour ago
    },
    {
      id: 'agno-1',
      name: 'Agno Document Processor',
      type: 'agno',
      status: 'online',
      description: 'Advanced document processing and analysis with OCR capabilities',
      capabilities: ['Document Analysis', 'OCR', 'Text Extraction', 'Summarization'],
      responseTime: 250,
      isActive: true,
      version: '1.0.0',
      lastUsed: new Date(Date.now() - 7200000), // 2 hours ago
    },
    {
      id: 'mastra-1',
      name: 'Mastra Workflow Engine',
      type: 'mastra',
      status: 'online',
      description: 'Workflow automation and business process management',
      capabilities: ['Workflow Automation', 'Task Management', 'Integration', 'Monitoring'],
      responseTime: 180,
      isActive: true,
      version: '1.0.0',
      lastUsed: new Date(Date.now() - 1800000), // 30 minutes ago
    },
  ];
}