/**
 * Approval Card Component
 * HITL (Human-in-the-Loop) approval interface for critical RAN operations
 *
 * @module ui/components/ApprovalCard
 * @version 7.0.0-alpha.1
 */

import React, { useState } from 'react';
import type { ApprovalRequest, SafetyCheck, ParameterChange } from '../types.js';

// ============================================================================
// Types
// ============================================================================

interface ApprovalCardProps {
  approval: ApprovalRequest;
  onApprove: (approvalId: string, signature: string, notes?: string) => void;
  onReject: (approvalId: string, signature: string, notes?: string) => void;
  compact?: boolean;
}

// ============================================================================
// Helper Functions
// ============================================================================

function getRiskColor(riskLevel: ApprovalRequest['risk_level']): string {
  const colors = {
    low: '#2ECC71',
    medium: '#FFA500',
    high: '#E74C3C',
    critical: '#8B0000'
  };
  return colors[riskLevel] || '#999';
}

function getSeverityIcon(severity: SafetyCheck['severity']): string {
  const icons = {
    info: 'ℹ️',
    warning: '⚠️',
    error: '❌'
  };
  return icons[severity] || '•';
}

function formatTimeRemaining(expiresAt: string): string {
  const now = new Date();
  const expiry = new Date(expiresAt);
  const diffMs = expiry.getTime() - now.getTime();

  if (diffMs < 0) {
    return 'EXPIRED';
  }

  const diffMins = Math.floor(diffMs / 60000);
  if (diffMins < 60) {
    return `${diffMins}m`;
  }

  const diffHours = Math.floor(diffMins / 60);
  return `${diffHours}h ${diffMins % 60}m`;
}

// ============================================================================
// Component
// ============================================================================

export const ApprovalCard: React.FC<ApprovalCardProps> = ({
  approval,
  onApprove,
  onReject,
  compact = false
}) => {
  const [signature, setSignature] = useState('');
  const [notes, setNotes] = useState('');
  const [showDetails, setShowDetails] = useState(!compact);
  const [isProcessing, setIsProcessing] = useState(false);

  const riskColor = getRiskColor(approval.risk_level);
  const timeRemaining = formatTimeRemaining(approval.expires_at);
  const isExpired = timeRemaining === 'EXPIRED';

  const handleApprove = async () => {
    if (!signature.trim()) {
      alert('Signature is required for approval');
      return;
    }

    setIsProcessing(true);
    try {
      await onApprove(approval.id, signature, notes || undefined);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReject = async () => {
    if (!signature.trim()) {
      alert('Signature is required for rejection');
      return;
    }

    setIsProcessing(true);
    try {
      await onReject(approval.id, signature, notes || undefined);
    } finally {
      setIsProcessing(false);
    }
  };

  const passedChecks = approval.safety_checks.filter(c => c.passed).length;
  const totalChecks = approval.safety_checks.length;
  const allChecksPassed = passedChecks === totalChecks;

  return (
    <div
      style={{
        border: `2px solid ${riskColor}`,
        borderRadius: '8px',
        background: '#fff',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
        marginBottom: '16px',
        opacity: isExpired ? 0.6 : 1
      }}
    >
      {/* Header */}
      <div
        style={{
          background: riskColor,
          color: '#fff',
          padding: '12px 16px',
          borderRadius: '6px 6px 0 0',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}
      >
        <div>
          <div style={{ fontSize: '16px', fontWeight: 'bold' }}>
            {approval.risk_level.toUpperCase()} RISK APPROVAL REQUIRED
          </div>
          <div style={{ fontSize: '12px', opacity: 0.9, marginTop: '4px' }}>
            ID: {approval.id}
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '14px', fontWeight: 'bold' }}>
            {isExpired ? '⏰ EXPIRED' : `⏱️ ${timeRemaining}`}
          </div>
          <div style={{ fontSize: '11px', opacity: 0.9 }}>
            {new Date(approval.created_at).toLocaleString()}
          </div>
        </div>
      </div>

      {/* Content */}
      <div style={{ padding: '16px' }}>
        {/* Action Description */}
        <div style={{ marginBottom: '12px' }}>
          <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '6px' }}>
            Requested Action:
          </div>
          <div
            style={{
              fontSize: '15px',
              padding: '10px',
              background: '#f5f5f5',
              borderRadius: '4px',
              border: '1px solid #e0e0e0'
            }}
          >
            {approval.action}
          </div>
        </div>

        {/* Target Cells */}
        <div style={{ marginBottom: '12px' }}>
          <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '6px' }}>
            Target Cells ({approval.target.length}):
          </div>
          <div style={{ fontSize: '13px', color: '#666', fontFamily: 'monospace' }}>
            {approval.target.join(', ')}
          </div>
        </div>

        {/* Justification */}
        <div style={{ marginBottom: '12px' }}>
          <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '6px' }}>
            Justification:
          </div>
          <div style={{ fontSize: '13px', color: '#333', lineHeight: '1.5' }}>
            {approval.justification}
          </div>
        </div>

        {/* Predicted Impact */}
        <div style={{ marginBottom: '12px' }}>
          <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '6px' }}>
            Predicted Impact:
          </div>
          <div style={{ fontSize: '13px', color: '#666', fontStyle: 'italic' }}>
            {approval.predicted_impact}
          </div>
        </div>

        {/* Safety Checks Summary */}
        <div style={{ marginBottom: '12px' }}>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              cursor: 'pointer',
              padding: '8px',
              background: allChecksPassed ? '#e8f5e9' : '#fff3e0',
              borderRadius: '4px',
              border: `1px solid ${allChecksPassed ? '#4CAF50' : '#FF9800'}`
            }}
            onClick={() => setShowDetails(!showDetails)}
          >
            <div style={{ fontSize: '14px', fontWeight: 'bold' }}>
              Safety Checks: {passedChecks}/{totalChecks} Passed
            </div>
            <div style={{ fontSize: '18px' }}>{showDetails ? '▼' : '▶'}</div>
          </div>

          {showDetails && (
            <div style={{ marginTop: '8px', paddingLeft: '12px' }}>
              {approval.safety_checks.map((check, idx) => (
                <div
                  key={idx}
                  style={{
                    padding: '6px 0',
                    borderBottom: idx < approval.safety_checks.length - 1 ? '1px solid #e0e0e0' : 'none',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px'
                  }}
                >
                  <span style={{ fontSize: '16px' }}>
                    {check.passed ? '✅' : getSeverityIcon(check.severity)}
                  </span>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '13px', fontWeight: 'bold' }}>{check.check_name}</div>
                    <div style={{ fontSize: '12px', color: '#666' }}>{check.message}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Proposed Changes */}
        {showDetails && approval.proposed_changes.length > 0 && (
          <div style={{ marginBottom: '12px' }}>
            <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '6px' }}>
              Proposed Parameter Changes:
            </div>
            <table style={{ width: '100%', fontSize: '12px', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: '#f5f5f5', borderBottom: '2px solid #ddd' }}>
                  <th style={{ padding: '8px', textAlign: 'left' }}>Cell</th>
                  <th style={{ padding: '8px', textAlign: 'left' }}>Parameter</th>
                  <th style={{ padding: '8px', textAlign: 'right' }}>Current</th>
                  <th style={{ padding: '8px', textAlign: 'right' }}>Proposed</th>
                  <th style={{ padding: '8px', textAlign: 'right' }}>Delta</th>
                  <th style={{ padding: '8px', textAlign: 'center' }}>Bounds</th>
                </tr>
              </thead>
              <tbody>
                {approval.proposed_changes.map((change, idx) => {
                  const delta = change.new_value - change.old_value;
                  const inBounds =
                    change.new_value >= change.bounds[0] && change.new_value <= change.bounds[1];

                  return (
                    <tr key={idx} style={{ borderBottom: '1px solid #e0e0e0' }}>
                      <td style={{ padding: '8px', fontFamily: 'monospace' }}>{change.cell_id}</td>
                      <td style={{ padding: '8px' }}>{change.parameter}</td>
                      <td style={{ padding: '8px', textAlign: 'right', fontFamily: 'monospace' }}>
                        {change.old_value.toFixed(2)}
                      </td>
                      <td
                        style={{
                          padding: '8px',
                          textAlign: 'right',
                          fontFamily: 'monospace',
                          fontWeight: 'bold',
                          color: inBounds ? '#333' : '#E74C3C'
                        }}
                      >
                        {change.new_value.toFixed(2)}
                      </td>
                      <td
                        style={{
                          padding: '8px',
                          textAlign: 'right',
                          fontFamily: 'monospace',
                          color: delta > 0 ? '#2ECC71' : delta < 0 ? '#E74C3C' : '#999'
                        }}
                      >
                        {delta > 0 ? '+' : ''}
                        {delta.toFixed(2)}
                      </td>
                      <td
                        style={{
                          padding: '8px',
                          textAlign: 'center',
                          fontSize: '11px',
                          color: '#666'
                        }}
                      >
                        [{change.bounds[0]}, {change.bounds[1]}]
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Approval Interface */}
        {approval.status === 'pending' && !isExpired && (
          <div
            style={{
              marginTop: '16px',
              padding: '16px',
              background: '#f9f9f9',
              borderRadius: '4px',
              border: '1px solid #e0e0e0'
            }}
          >
            <div style={{ marginBottom: '12px' }}>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: 'bold', marginBottom: '6px' }}>
                Operator Signature (required):
              </label>
              <input
                type="text"
                value={signature}
                onChange={e => setSignature(e.target.value)}
                placeholder="Enter your operator ID or signature"
                style={{
                  width: '100%',
                  padding: '8px',
                  fontSize: '14px',
                  border: '1px solid #ccc',
                  borderRadius: '4px',
                  boxSizing: 'border-box'
                }}
                disabled={isProcessing}
              />
            </div>

            <div style={{ marginBottom: '12px' }}>
              <label style={{ display: 'block', fontSize: '14px', fontWeight: 'bold', marginBottom: '6px' }}>
                Notes (optional):
              </label>
              <textarea
                value={notes}
                onChange={e => setNotes(e.target.value)}
                placeholder="Add any notes or comments..."
                rows={3}
                style={{
                  width: '100%',
                  padding: '8px',
                  fontSize: '14px',
                  border: '1px solid #ccc',
                  borderRadius: '4px',
                  boxSizing: 'border-box',
                  resize: 'vertical'
                }}
                disabled={isProcessing}
              />
            </div>

            <div style={{ display: 'flex', gap: '12px' }}>
              <button
                onClick={handleApprove}
                disabled={isProcessing || !signature.trim()}
                style={{
                  flex: 1,
                  padding: '12px',
                  fontSize: '16px',
                  fontWeight: 'bold',
                  background: '#2ECC71',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: isProcessing || !signature.trim() ? 'not-allowed' : 'pointer',
                  opacity: isProcessing || !signature.trim() ? 0.6 : 1
                }}
              >
                {isProcessing ? '⏳ Processing...' : '✓ APPROVE'}
              </button>

              <button
                onClick={handleReject}
                disabled={isProcessing || !signature.trim()}
                style={{
                  flex: 1,
                  padding: '12px',
                  fontSize: '16px',
                  fontWeight: 'bold',
                  background: '#E74C3C',
                  color: '#fff',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: isProcessing || !signature.trim() ? 'not-allowed' : 'pointer',
                  opacity: isProcessing || !signature.trim() ? 0.6 : 1
                }}
              >
                {isProcessing ? '⏳ Processing...' : '✗ REJECT'}
              </button>
            </div>
          </div>
        )}

        {/* Status Display */}
        {approval.status !== 'pending' && (
          <div
            style={{
              marginTop: '16px',
              padding: '12px',
              background: approval.status === 'approved' ? '#e8f5e9' : '#ffebee',
              border: `2px solid ${approval.status === 'approved' ? '#4CAF50' : '#E74C3C'}`,
              borderRadius: '4px',
              textAlign: 'center',
              fontSize: '16px',
              fontWeight: 'bold',
              color: approval.status === 'approved' ? '#2E7D32' : '#C62828'
            }}
          >
            {approval.status === 'approved' ? '✓ APPROVED' : '✗ REJECTED'}
          </div>
        )}

        {isExpired && approval.status === 'pending' && (
          <div
            style={{
              marginTop: '16px',
              padding: '12px',
              background: '#fff3e0',
              border: '2px solid #FF9800',
              borderRadius: '4px',
              textAlign: 'center',
              fontSize: '16px',
              fontWeight: 'bold',
              color: '#E65100'
            }}
          >
            ⏰ APPROVAL EXPIRED
          </div>
        )}
      </div>
    </div>
  );
};

// ============================================================================
// Export
// ============================================================================

export default ApprovalCard;
