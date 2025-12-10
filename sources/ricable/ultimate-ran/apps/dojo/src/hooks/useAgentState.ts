/**
 * useAgentState Hook - Connects to ruvector stream for real-time council updates
 * Part of Ericsson Gen 7.0 Neuro-Symbolic Titan Platform
 */

import { useState, useEffect, useCallback } from 'react';
import type { AgentState, ThinkingStepEvent, ApprovalRequest, InterferenceData } from '../types/council';

/**
 * Mock stream connection - In production, this would connect to ruvector QUIC stream
 * via agentic-flow protocol
 */
class RuvectorStream {
  private listeners: Set<(event: any) => void> = new Set();
  private mockInterval: NodeJS.Timeout | null = null;

  connect() {
    // Simulate periodic events for demonstration
    // In production: connect to ws://ruvector-service/stream
    console.log('[RuvectorStream] Connected to council stream');
  }

  subscribe(callback: (event: any) => void) {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  emit(event: any) {
    this.listeners.forEach(listener => listener(event));
  }

  disconnect() {
    if (this.mockInterval) {
      clearInterval(this.mockInterval);
    }
    this.listeners.clear();
    console.log('[RuvectorStream] Disconnected from council stream');
  }
}

const stream = new RuvectorStream();

export function useAgentState(): AgentState & {
  approveRequest: (id: string) => void;
  rejectRequest: (id: string) => void;
} {
  const [state, setState] = useState<AgentState>({
    councilMembers: [
      {
        id: 'analyst-deepseek',
        role: 'analyst',
        model_id: 'deepseek-r1-distill',
        temperature: 0.2,
        avatar: '🔬'
      },
      {
        id: 'historian-gemini',
        role: 'historian',
        model_id: 'gemini-1.5-pro',
        temperature: 0.7,
        avatar: '📚'
      },
      {
        id: 'strategist-claude',
        role: 'strategist',
        model_id: 'claude-3-7-sonnet',
        temperature: 0.5,
        avatar: '⚔️'
      },
      {
        id: 'chairman',
        role: 'chairman',
        model_id: 'claude-3-7-sonnet',
        temperature: 0.3,
        avatar: '⚖️'
      }
    ],
    debateHistory: [],
    currentInterference: null,
    pendingApprovals: [],
    isDebating: false
  });

  useEffect(() => {
    stream.connect();

    const unsubscribe = stream.subscribe((event: any) => {
      if (event.type === 'THINKING_STEP') {
        setState(prev => ({
          ...prev,
          debateHistory: [...prev.debateHistory, event as ThinkingStepEvent],
          isDebating: true
        }));
      } else if (event.type === 'INTERFERENCE_UPDATE') {
        setState(prev => ({
          ...prev,
          currentInterference: event.data as InterferenceData
        }));
      } else if (event.type === 'APPROVAL_REQUEST') {
        setState(prev => ({
          ...prev,
          pendingApprovals: [...prev.pendingApprovals, event.data as ApprovalRequest]
        }));
      } else if (event.type === 'DEBATE_COMPLETE') {
        setState(prev => ({ ...prev, isDebating: false }));
      }
    });

    return () => {
      unsubscribe();
      stream.disconnect();
    };
  }, []);

  const approveRequest = useCallback((id: string) => {
    setState(prev => ({
      ...prev,
      pendingApprovals: prev.pendingApprovals.filter(req => req.id !== id)
    }));
    console.log(`[HITL] Approved request ${id}`);
    // In production: emit approval to council via agentic-flow
  }, []);

  const rejectRequest = useCallback((id: string) => {
    setState(prev => ({
      ...prev,
      pendingApprovals: prev.pendingApprovals.filter(req => req.id !== id)
    }));
    console.log(`[HITL] Rejected request ${id}`);
    // In production: emit rejection to council via agentic-flow
  }, []);

  return {
    ...state,
    approveRequest,
    rejectRequest
  };
}

// Export stream for testing/debugging
export { stream };
