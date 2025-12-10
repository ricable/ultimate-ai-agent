// File: frontend/src/components/collaboration/CollaborationDashboard.tsx
import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardContent } from '../ui/Card';
import { UserCollaborationProfile } from './UserCollaborationProfile';
import { TrustCalibrationPanel } from './TrustCalibrationPanel';
import { CognitiveLoadMonitor } from './CognitiveLoadMonitor';
import { ContextAwarenessPanel } from './ContextAwarenessPanel';
import { AdaptiveInterfacePanel } from './AdaptiveInterfacePanel';
import { ExplanationPanel } from './ExplanationPanel';

interface CollaborationSession {
  session_id: string;
  user_id: string;
  ai_agent_id: string;
  collaboration_mode: string;
  task_description: string;
  status: string;
  started_at: string;
  trust_metrics: {
    current_trust: number;
    trust_trend: number;
  };
  cognitive_load_data: any[];
  adaptation_log: any[];
}

interface CollaborationDashboardProps {
  userId: string;
  agentId: string;
}

export function CollaborationDashboard({ userId, agentId }: CollaborationDashboardProps) {
  const [currentSession, setCurrentSession] = useState<CollaborationSession | null>(null);
  const [userProfile, setUserProfile] = useState<any>(null);
  const [trustStatus, setTrustStatus] = useState<any>(null);
  const [cognitiveLoad, setCognitiveLoad] = useState<any>(null);
  const [contextData, setContextData] = useState<any>(null);
  const [adaptations, setAdaptations] = useState<any[]>([]);
  const [explanations, setExplanations] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // API endpoints
  const API_BASE = 'http://localhost:8000';

  useEffect(() => {
    initializeCollaboration();
    // Set up real-time updates
    const interval = setInterval(updateCollaborationData, 5000);
    return () => clearInterval(interval);
  }, [userId, agentId]);

  const initializeCollaboration = async () => {
    try {
      setIsLoading(true);
      
      // Start collaboration session
      const sessionResponse = await fetch(`${API_BASE}/api/collaboration/start-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          ai_agent_id: agentId,
          task_description: 'General collaboration session',
          initial_context: {
            task_type: 'general',
            complexity: 'medium',
            domain: 'assistant'
          }
        })
      });

      if (!sessionResponse.ok) {
        throw new Error('Failed to start collaboration session');
      }

      const sessionData = await sessionResponse.json();
      setCurrentSession(sessionData);

      // Load initial data
      await Promise.all([
        loadUserProfile(),
        loadTrustStatus(),
        loadCognitiveLoad(),
        loadContextData()
      ]);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to initialize collaboration');
    } finally {
      setIsLoading(false);
    }
  };

  const loadUserProfile = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/collaboration/user-profile/${userId}`);
      if (response.ok) {
        const profile = await response.json();
        setUserProfile(profile);
      }
    } catch (err) {
      console.error('Failed to load user profile:', err);
    }
  };

  const loadTrustStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/trust/user-status/${userId}`);
      if (response.ok) {
        const trust = await response.json();
        setTrustStatus(trust);
      }
    } catch (err) {
      console.error('Failed to load trust status:', err);
    }
  };

  const loadCognitiveLoad = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/cognitive-load/user-status/${userId}`);
      if (response.ok) {
        const load = await response.json();
        setCognitiveLoad(load);
      }
    } catch (err) {
      console.error('Failed to load cognitive load:', err);
    }
  };

  const loadContextData = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/context/current/${userId}`);
      if (response.ok) {
        const context = await response.json();
        setContextData(context);
      }
    } catch (err) {
      console.error('Failed to load context data:', err);
    }
  };

  const updateCollaborationData = async () => {
    if (!currentSession) return;

    try {
      // Update session status
      const sessionResponse = await fetch(`${API_BASE}/api/collaboration/session-status/${currentSession.session_id}`);
      if (sessionResponse.ok) {
        const sessionData = await sessionResponse.json();
        setCurrentSession(sessionData);
      }

      // Update other data
      await Promise.all([
        loadTrustStatus(),
        loadCognitiveLoad(),
        loadContextData()
      ]);

    } catch (err) {
      console.error('Failed to update collaboration data:', err);
    }
  };

  const handleInteraction = async (interactionType: string, data: any) => {
    if (!currentSession) return;

    try {
      const response = await fetch(`${API_BASE}/api/collaboration/interaction`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: currentSession.session_id,
          interaction_type: interactionType,
          interaction_data: data,
          context: contextData
        })
      });

      if (response.ok) {
        const result = await response.json();
        
        // Update adaptations if provided
        if (result.adaptations) {
          setAdaptations(prev => [...prev, ...result.adaptations]);
        }

        // Trigger data refresh
        updateCollaborationData();
      }
    } catch (err) {
      console.error('Failed to process interaction:', err);
    }
  };

  const requestExplanation = async (target: string, type: string = 'causal') => {
    if (!currentSession) return;

    try {
      const response = await fetch(`${API_BASE}/api/explanation/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          session_id: currentSession.session_id,
          explanation_target: target,
          preferred_type: type,
          context: contextData,
          user_profile: userProfile
        })
      });

      if (response.ok) {
        const explanation = await response.json();
        setExplanations(prev => [explanation, ...prev]);
      }
    } catch (err) {
      console.error('Failed to generate explanation:', err);
    }
  };

  const handleTrustFeedback = async (feedbackType: string, rating: number) => {
    if (!currentSession) return;

    try {
      await fetch(`${API_BASE}/api/trust/record-event`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          session_id: currentSession.session_id,
          event_type: feedbackType,
          event_data: { rating },
          context: contextData
        })
      });

      // Refresh trust data
      loadTrustStatus();
    } catch (err) {
      console.error('Failed to record trust feedback:', err);
    }
  };

  const measureCognitiveLoad = async (interactionData: any) => {
    if (!currentSession) return;

    try {
      await fetch(`${API_BASE}/api/cognitive-load/measure`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          session_id: currentSession.session_id,
          interaction_data: interactionData,
          context: contextData
        })
      });

      // Refresh cognitive load data
      loadCognitiveLoad();
    } catch (err) {
      console.error('Failed to measure cognitive load:', err);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-600">Initializing collaboration session...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8">
        <Card className="border-red-200 bg-red-50">
          <CardContent className="p-6">
            <h3 className="text-red-800 font-semibold mb-2">Collaboration Error</h3>
            <p className="text-red-700">{error}</p>
            <button
              onClick={initializeCollaboration}
              className="mt-4 bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
            >
              Retry Initialization
            </button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6 rounded-lg">
        <h1 className="text-2xl font-bold mb-2">Human-AI Collaboration Dashboard</h1>
        <p className="opacity-90">
          Advanced collaborative intelligence with context awareness and adaptive interfaces
        </p>
        {currentSession && (
          <div className="mt-4 flex items-center space-x-4 text-sm">
            <span>Session: {currentSession.session_id.slice(0, 8)}...</span>
            <span>Mode: {currentSession.collaboration_mode}</span>
            <span className={`px-2 py-1 rounded ${
              currentSession.status === 'active' ? 'bg-green-500' : 'bg-gray-500'
            }`}>
              {currentSession.status}
            </span>
          </div>
        )}
      </div>

      {/* Main Dashboard Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        
        {/* User Profile & Session Info */}
        <div className="space-y-6">
          <UserCollaborationProfile
            userProfile={userProfile}
            currentSession={currentSession}
            onProfileUpdate={loadUserProfile}
          />
          
          <TrustCalibrationPanel
            trustStatus={trustStatus}
            onTrustFeedback={handleTrustFeedback}
            onRequestExplanation={() => requestExplanation('trust_level', 'causal')}
          />
        </div>

        {/* Cognitive Load & Context */}
        <div className="space-y-6">
          <CognitiveLoadMonitor
            cognitiveLoad={cognitiveLoad}
            onLoadMeasurement={measureCognitiveLoad}
            adaptations={adaptations}
          />
          
          <ContextAwarenessPanel
            contextData={contextData}
            onContextUpdate={loadContextData}
            onInteraction={handleInteraction}
          />
        </div>

        {/* Adaptations & Explanations */}
        <div className="space-y-6">
          <AdaptiveInterfacePanel
            adaptations={adaptations}
            cognitiveLoad={cognitiveLoad}
            trustStatus={trustStatus}
            onApplyAdaptation={(adaptation) => handleInteraction('apply_adaptation', adaptation)}
          />
          
          <ExplanationPanel
            explanations={explanations}
            onRequestExplanation={requestExplanation}
            onExplanationFeedback={async (explanationId, feedback) => {
              try {
                await fetch(`${API_BASE}/api/explanation/feedback`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({
                    explanation_id: explanationId,
                    user_id: userId,
                    ...feedback
                  })
                });
              } catch (err) {
                console.error('Failed to record explanation feedback:', err);
              }
            }}
          />
        </div>
      </div>

      {/* Real-time Status Bar */}
      <div className="fixed bottom-4 right-4 bg-white border border-gray-200 rounded-lg shadow-lg p-4 min-w-64">
        <h4 className="font-semibold text-sm text-gray-700 mb-2">Collaboration Status</h4>
        <div className="space-y-2 text-xs">
          {trustStatus && (
            <div className="flex justify-between">
              <span>Trust Level:</span>
              <span className={`font-medium ${
                trustStatus.overall_trust > 0.7 ? 'text-green-600' :
                trustStatus.overall_trust > 0.4 ? 'text-yellow-600' : 'text-red-600'
              }`}>
                {(trustStatus.overall_trust * 100).toFixed(0)}%
              </span>
            </div>
          )}
          {cognitiveLoad && (
            <div className="flex justify-between">
              <span>Cognitive Load:</span>
              <span className={`font-medium ${
                cognitiveLoad.current_status?.load_level === 'low' ? 'text-green-600' :
                cognitiveLoad.current_status?.load_level === 'moderate' ? 'text-yellow-600' : 'text-red-600'
              }`}>
                {cognitiveLoad.current_status?.load_level || 'unknown'}
              </span>
            </div>
          )}
          <div className="flex justify-between">
            <span>Adaptations:</span>
            <span className="font-medium text-blue-600">{adaptations.length} active</span>
          </div>
        </div>
      </div>
    </div>
  );
}