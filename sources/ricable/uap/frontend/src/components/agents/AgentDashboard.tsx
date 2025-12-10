// File: frontend/src/components/agents/AgentDashboard.tsx
import { AgentCard } from './AgentCard';
import { useAuth } from '../../auth/AuthContext';
import { useEffect, useState } from 'react';
import { apiConfig } from '../../lib/api-config';

interface Agent {
  id: string;
  name: string;
  description: string;
  framework: string;
  status?: string;
  performance?: any;
}

export function AgentDashboard() {
  const { token } = useAuth();
  const [agents, setAgents] = useState<Agent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAgents = async () => {
      try {
        setLoading(true);
        setError(null);

        const headers = apiConfig.createHeaders(token);

        const response = await fetch(apiConfig.getEndpoint('/api/agents'), {
          method: 'GET',
          headers,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const agentData = await response.json();
        setAgents(agentData);
      } catch (err) {
        console.error('Failed to fetch agents:', err);
        setError('Failed to load agents. Please try again.');
        
        // Fallback to default agents if API fails
        setAgents([
          {
            id: 'research-agent',
            name: 'Research Agent',
            description: 'Specializes in web searches and document analysis using Agno.',
            framework: 'agno',
          },
          {
            id: 'support-agent',
            name: 'Customer Support Agent',
            description: 'Handles customer queries with predefined workflows using Mastra.',
            framework: 'mastra',
          },
          {
            id: 'general-assistant',
            name: 'General Assistant',
            description: 'A general-purpose assistant powered by CopilotKit.',
            framework: 'copilot',
          },
          {
            id: 'metacognition-agent',
            name: 'Metacognition Agent',
            description: 'Self-improving AI with metacognitive awareness, introspection, and safety constraints.',
            framework: 'metacognition',
          },
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchAgents();
  }, [token]);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-6">
        <div className="text-gray-600">Loading agents...</div>
      </div>
    );
  }

  if (error && agents.length === 0) {
    return (
      <div className="flex items-center justify-center p-6">
        <div className="text-red-600">{error}</div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {error && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
          <div className="text-yellow-800">
            Warning: {error} Showing fallback data.
          </div>
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 p-6">
        {agents.map((agent) => (
          <AgentCard
            key={agent.id}
            id={agent.id}
            name={agent.name}
            description={agent.description}
            framework={agent.framework}
          />
        ))}
      </div>
    </div>
  );
}