/**
 * ApprovalCard - HITL (Human-in-the-Loop) Approval Interface
 * Part of Ericsson Gen 7.0 Neuro-Symbolic Titan Platform
 *
 * The "Chairman's Gavel" - Final human approval for critical actions.
 * Displays when council cannot reach 2/3+ consensus or action is high-risk.
 */

'use client';

import React, { useState } from 'react';
import type { ApprovalRequest } from '../../types/council';

interface ApprovalCardProps {
  request: ApprovalRequest;
  onApprove: (id: string) => void;
  onReject: (id: string) => void;
  className?: string;
}

const riskColors = {
  low: 'border-green-500 bg-green-500/10',
  medium: 'border-yellow-500 bg-yellow-500/10',
  high: 'border-orange-500 bg-orange-500/10',
  critical: 'border-red-500 bg-red-500/10'
};

const riskIcons = {
  low: '✅',
  medium: '⚠️',
  high: '🔥',
  critical: '🚨'
};

const typeLabels = {
  parameter_change: 'Parameter Change',
  reboot: 'Cell Reboot',
  topology_change: 'Topology Change'
};

export function ApprovalCard({
  request,
  onApprove,
  onReject,
  className = ''
}: ApprovalCardProps) {
  const [isProcessing, setIsProcessing] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  const handleApprove = async () => {
    setIsProcessing(true);
    // Simulate async approval process
    await new Promise(resolve => setTimeout(resolve, 500));
    onApprove(request.id);
    setIsProcessing(false);
  };

  const handleReject = async () => {
    setIsProcessing(true);
    // Simulate async rejection process
    await new Promise(resolve => setTimeout(resolve, 500));
    onReject(request.id);
    setIsProcessing(false);
  };

  const consensusPercentage = Math.round(request.consensusScore * 100);
  const needsHumanApproval = request.consensusScore < 0.67 || request.risk === 'critical';

  return (
    <div className={`relative ${className}`}>
      {/* Glass Box Container with Risk-based Border */}
      <div className={`bg-gray-900/50 backdrop-blur-sm rounded-lg border-2 p-4 ${riskColors[request.risk]}`}>
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center gap-3">
            <span className="text-3xl">{riskIcons[request.risk]}</span>
            <div>
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                {typeLabels[request.type]}
                {needsHumanApproval && (
                  <span className="text-xs px-2 py-1 bg-red-500/20 text-red-300 rounded border border-red-500">
                    HITL Required
                  </span>
                )}
              </h3>
              <p className="text-sm text-gray-400">
                Proposed by: {request.proposedBy.role} ({request.proposedBy.model_id})
              </p>
            </div>
          </div>
          <div className="text-right">
            <div className="text-xs text-gray-400">Risk Level</div>
            <div className={`text-lg font-bold uppercase ${
              request.risk === 'critical' ? 'text-red-400' :
              request.risk === 'high' ? 'text-orange-400' :
              request.risk === 'medium' ? 'text-yellow-400' :
              'text-green-400'
            }`}>
              {request.risk}
            </div>
          </div>
        </div>

        {/* Description */}
        <div className="mb-4 p-3 bg-gray-800/50 rounded border border-gray-700">
          <p className="text-sm text-gray-100 leading-relaxed">
            {request.description}
          </p>
        </div>

        {/* Consensus Meter */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-400">Council Consensus</span>
            <span className={`text-sm font-semibold ${
              request.consensusScore >= 0.67 ? 'text-green-400' : 'text-yellow-400'
            }`}>
              {consensusPercentage}%
            </span>
          </div>
          <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-500 ${
                request.consensusScore >= 0.67 ? 'bg-green-500' : 'bg-yellow-500'
              }`}
              style={{ width: `${consensusPercentage}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>0%</span>
            <span className="text-gray-400">67% (2/3 threshold)</span>
            <span>100%</span>
          </div>
        </div>

        {/* Parameters Toggle */}
        <div className="mb-4">
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="text-sm text-blue-400 hover:text-blue-300 flex items-center gap-2"
          >
            <span>{showDetails ? '▼' : '▶'}</span>
            {showDetails ? 'Hide' : 'Show'} Parameters
          </button>
          {showDetails && (
            <div className="mt-2 p-3 bg-gray-900/50 rounded border border-gray-700">
              <pre className="text-xs text-gray-300 overflow-x-auto">
                {JSON.stringify(request.parameters, null, 2)}
              </pre>
            </div>
          )}
        </div>

        {/* Action Buttons - The "Chairman's Gavel" */}
        <div className="flex gap-3">
          <button
            onClick={handleApprove}
            disabled={isProcessing}
            className={`flex-1 px-4 py-3 rounded-lg font-semibold transition-all ${
              isProcessing
                ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
                : 'bg-green-600 hover:bg-green-500 text-white shadow-lg shadow-green-900/50'
            }`}
          >
            {isProcessing ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Processing...
              </span>
            ) : (
              <span className="flex items-center justify-center gap-2">
                <span>⚖️</span>
                Approve Action
              </span>
            )}
          </button>

          <button
            onClick={handleReject}
            disabled={isProcessing}
            className={`flex-1 px-4 py-3 rounded-lg font-semibold transition-all ${
              isProcessing
                ? 'bg-gray-700 text-gray-400 cursor-not-allowed'
                : 'bg-red-600 hover:bg-red-500 text-white shadow-lg shadow-red-900/50'
            }`}
          >
            {isProcessing ? (
              'Processing...'
            ) : (
              <span className="flex items-center justify-center gap-2">
                <span>🛑</span>
                Reject Action
              </span>
            )}
          </button>
        </div>

        {/* Timestamp */}
        <div className="mt-3 text-xs text-gray-500 text-center">
          Requested at {new Date(request.timestamp).toLocaleString()}
        </div>
      </div>
    </div>
  );
}
