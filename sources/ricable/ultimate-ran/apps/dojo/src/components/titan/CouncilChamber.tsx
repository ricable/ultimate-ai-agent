/**
 * CouncilChamber - Main "War Room" Container
 * Part of Ericsson Gen 7.0 Neuro-Symbolic Titan Platform
 *
 * The central interface for monitoring and controlling the LLM Council.
 * Integrates all components: Debate Timeline, Interference Heatmap, and HITL Approvals.
 */

'use client';

import React from 'react';
import { useAgentState } from '../../hooks/useAgentState';
import { InterferenceHeatmap } from './InterferenceHeatmap';
import { DebateTimeline } from './DebateTimeline';
import { ApprovalCard } from './ApprovalCard';

interface CouncilChamberProps {
  className?: string;
}

export function CouncilChamber({ className = '' }: CouncilChamberProps) {
  const {
    councilMembers,
    debateHistory,
    currentInterference,
    pendingApprovals,
    isDebating,
    approveRequest,
    rejectRequest
  } = useAgentState();

  return (
    <div className={`min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-black ${className}`}>
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="text-3xl">⚡</div>
              <div>
                <h1 className="text-2xl font-bold text-white">
                  Titan Council Chamber
                </h1>
                <p className="text-sm text-gray-400">
                  Gen 7.0 Neuro-Symbolic RAN Optimization
                </p>
              </div>
            </div>

            {/* Status Indicators */}
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 px-3 py-2 bg-gray-800 rounded-lg">
                <div className={`w-3 h-3 rounded-full ${
                  isDebating ? 'bg-green-500 animate-pulse' : 'bg-gray-600'
                }`} />
                <span className="text-sm text-gray-300">
                  {isDebating ? 'Active' : 'Idle'}
                </span>
              </div>
              <div className="flex items-center gap-2 px-3 py-2 bg-gray-800 rounded-lg">
                <span className="text-sm text-gray-400">Council Members:</span>
                <span className="text-sm font-semibold text-white">
                  {councilMembers.length}
                </span>
              </div>
              {pendingApprovals.length > 0 && (
                <div className="flex items-center gap-2 px-3 py-2 bg-red-500/20 border border-red-500 rounded-lg animate-pulse">
                  <span className="text-sm font-semibold text-red-300">
                    {pendingApprovals.length} Pending Approval{pendingApprovals.length !== 1 ? 's' : ''}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-6 py-8">
        {/* Pending Approvals Section */}
        {pendingApprovals.length > 0 && (
          <section className="mb-8">
            <h2 className="text-xl font-semibold text-white mb-4 flex items-center gap-2">
              <span>⚖️</span>
              Chairman's Gavel - HITL Approvals
            </h2>
            <div className="grid grid-cols-1 gap-4">
              {pendingApprovals.map(request => (
                <ApprovalCard
                  key={request.id}
                  request={request}
                  onApprove={approveRequest}
                  onReject={rejectRequest}
                />
              ))}
            </div>
          </section>
        )}

        {/* Main Grid Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column: Debate Timeline */}
          <section>
            <DebateTimeline
              events={debateHistory}
              councilMembers={councilMembers}
              isDebating={isDebating}
            />
          </section>

          {/* Right Column: Interference Visualization */}
          <section>
            <InterferenceHeatmap data={currentInterference} />

            {/* Additional Info Cards */}
            <div className="mt-6 space-y-4">
              {/* Quick Stats */}
              <div className="bg-gray-900/50 backdrop-blur-sm rounded-lg border border-gray-700 p-4">
                <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                  <span>📊</span>
                  Session Statistics
                </h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <div className="text-2xl font-bold text-blue-400">
                      {debateHistory.length}
                    </div>
                    <div className="text-sm text-gray-400">Total Events</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-green-400">
                      {debateHistory.filter(e => e.status === 'CONSENSUS').length}
                    </div>
                    <div className="text-sm text-gray-400">Consensus Reached</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-yellow-400">
                      {debateHistory.filter(e => e.status === 'CRITIQUE').length}
                    </div>
                    <div className="text-sm text-gray-400">Critiques</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold text-purple-400">
                      {debateHistory.filter(e => e.status === 'PROPOSAL').length}
                    </div>
                    <div className="text-sm text-gray-400">Proposals</div>
                  </div>
                </div>
              </div>

              {/* System Info */}
              <div className="bg-gray-900/50 backdrop-blur-sm rounded-lg border border-gray-700 p-4">
                <h3 className="text-lg font-semibold text-white mb-3 flex items-center gap-2">
                  <span>🔧</span>
                  System Configuration
                </h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Transport:</span>
                    <span className="text-gray-300 font-mono">agentic-flow (QUIC)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Memory:</span>
                    <span className="text-gray-300 font-mono">agentdb@alpha</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Topology:</span>
                    <span className="text-gray-300 font-mono">ruvector (mesh)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Security:</span>
                    <span className="text-gray-300 font-mono">ML-DSA-87</span>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* Footer Info */}
        <footer className="mt-8 pt-6 border-t border-gray-800 text-center text-sm text-gray-500">
          <p>
            Ericsson Gen 7.0 "Neuro-Symbolic Titan" Platform |
            LLM Council Architecture |
            SPARC Governance Enabled
          </p>
        </footer>
      </main>
    </div>
  );
}
