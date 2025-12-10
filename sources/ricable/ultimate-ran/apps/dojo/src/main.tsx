import React, { useEffect } from 'react';
import ReactDOM from 'react-dom/client';
import { CouncilChamber } from './components/titan/CouncilChamber';
import { stream } from './hooks/useAgentState';

// --- Mock Data Generator ---
const generateMockEvents = () => {
  // 1. Initial Interference Data
  setTimeout(() => {
    const matrixSize = 10;
    const matrix = Array(matrixSize).fill(0).map(() => Array(matrixSize).fill(0).map(() => -120 + Math.random() * 40));
    // Create a hotspot
    matrix[4][5] = -65;
    matrix[4][6] = -70;
    matrix[5][5] = -60;
    
    stream.emit({
      type: 'INTERFERENCE_UPDATE',
      data: {
        cellId: 'Cell-042-Alpha',
        matrix,
        neighbors: Array(matrixSize).fill(0).map((_, i) => ({ cellId: `N-${i}`, pci: 100+i, rsrp: -90 }))
      }
    });
  }, 1000);

  // 2. Debate Sequence
  const debateSequence = [
    {
      agentId: 'analyst-deepseek',
      agentName: 'Analyst (DeepSeek)',
      role: 'analyst',
      status: 'THINKING',
      content: 'Detected severe interference pattern in Sector 5 (Cell-042). RSRP degradation correlating with neighboring Cell-043 power spike.',
      metadata: { confidence: 0.98, source: 'telemetry_stream', correlation: 0.85 }
    },
    {
      agentId: 'historian-gemini',
      agentName: 'Historian (Gemini)',
      role: 'historian',
      status: 'CRITIQUE',
      content: 'Reviewing historical logs. Similar pattern observed 48 hours ago. Previous resolution involved down-tilting Cell-043 by 2 degrees.',
      metadata: { incident_id: 'INC-2025-892', success_rate: 0.92 }
    },
    {
      agentId: 'strategist-claude',
      agentName: 'Strategist (Claude)',
      role: 'strategist',
      status: 'PROPOSAL',
      content: 'Proposing coordinated action: 1) Down-tilt Cell-043 by 1.5° (conservative). 2) Reduce Tx Power on Cell-042 by 1dB to mitigate edge noise.',
      metadata: { strategy: 'interference_mitigation', risk_assessment: 'low' }
    },
    {
      agentId: 'chairman',
      agentName: 'Chairman',
      role: 'chairman',
      status: 'CONSENSUS',
      content: 'Proposal accepted for evaluation. Simulating coverage impact...',
      metadata: { consensus_score: 0.95 }
    },
    {
      agentId: 'analyst-deepseek',
      agentName: 'Analyst (DeepSeek)',
      role: 'analyst',
      status: 'THINKING',
      content: 'Simulation complete. Coverage hole probability < 2%. Throughput improvement estimated at +15%.',
      metadata: { sim_engine: 'ruvector_v2', latency: '45ms' }
    }
  ];

  let delay = 2000;
  debateSequence.forEach(event => {
    setTimeout(() => {
      stream.emit({
        type: 'THINKING_STEP',
        ...event,
        timestamp: new Date().toISOString()
      });
    }, delay);
    delay += 2500;
  });

  // 3. Approval Request (Critical)
  setTimeout(() => {
    stream.emit({
      type: 'APPROVAL_REQUEST',
      data: {
        id: 'REQ-9982',
        type: 'parameter_change',
        risk: 'high',
        description: 'AUTOMATED PROPOSAL: Adjust physical antenna tilt for Cell-043. Mechanical downtilt actuator engagement required.',
        proposedBy: { role: 'strategist', model_id: 'claude-3-7-sonnet' },
        consensusScore: 0.88,
        parameters: { cell_id: 'Cell-043', parameter: 'mechanical_tilt', old: 4, new: 6 },
        timestamp: new Date().toISOString()
      }
    });
  }, 15000);
  
    // 3b. Another Approval Request (Medium)
  setTimeout(() => {
    stream.emit({
      type: 'APPROVAL_REQUEST',
      data: {
        id: 'REQ-9983',
        type: 'topology_change',
        risk: 'medium',
        description: 'Neighbor Relation Update: Blacklist Cell-043 <-> Cell-099 due to ping-pong handover loops.',
        proposedBy: { role: 'historian', model_id: 'gemini-1.5-pro' },
        consensusScore: 0.72,
        parameters: { source: 'Cell-043', target: 'Cell-099', action: 'blacklist' },
        timestamp: new Date().toISOString()
      }
    });
  }, 16000);
};

// --- App Component ---
function App() {
  useEffect(() => {
    generateMockEvents();
  }, []);

  return <CouncilChamber />;
}

// --- Mount ---
const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
