/**
 * Optimization Timeline Component
 * Timeline visualization of GNN decisions and council debates
 *
 * @module ui/components/OptimizationTimeline
 * @version 7.0.0-alpha.1
 */

import React, { useState, useEffect } from 'react';
import type { OptimizationEvent } from '../types.js';

// ============================================================================
// Types
// ============================================================================

interface OptimizationTimelineProps {
  events: OptimizationEvent[];
  maxEvents?: number;
  onEventClick?: (event: OptimizationEvent) => void;
  autoScroll?: boolean;
}

interface TimelineGroup {
  date: string;
  events: OptimizationEvent[];
}

// ============================================================================
// Helper Functions
// ============================================================================

function getEventColor(eventType: OptimizationEvent['event_type']): string {
  const colors = {
    gnn_decision: '#4A90E2',
    council_debate: '#7B68EE',
    hitl_approval: '#FFA500',
    execution: '#50C878',
    rollback: '#E74C3C'
  };
  return colors[eventType] || '#999';
}

function getStatusBadgeColor(status: OptimizationEvent['status']): string {
  const colors = {
    pending: '#FFA500',
    approved: '#50C878',
    rejected: '#E74C3C',
    executed: '#2ECC71',
    rolled_back: '#95A5A6'
  };
  return colors[status] || '#999';
}

function formatTimestamp(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
}

function formatDate(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  });
}

function groupEventsByDate(events: OptimizationEvent[]): TimelineGroup[] {
  const groups = new Map<string, OptimizationEvent[]>();

  for (const event of events) {
    const date = formatDate(event.timestamp);
    if (!groups.has(date)) {
      groups.set(date, []);
    }
    groups.get(date)!.push(event);
  }

  return Array.from(groups.entries())
    .map(([date, events]) => ({ date, events }))
    .sort((a, b) => new Date(b.events[0].timestamp).getTime() - new Date(a.events[0].timestamp).getTime());
}

// ============================================================================
// Component
// ============================================================================

export const OptimizationTimeline: React.FC<OptimizationTimelineProps> = ({
  events,
  maxEvents = 100,
  onEventClick,
  autoScroll = true
}) => {
  const [expandedEvents, setExpandedEvents] = useState<Set<string>>(new Set());
  const [filterType, setFilterType] = useState<string>('all');
  const [filterStatus, setFilterStatus] = useState<string>('all');

  const filteredEvents = events
    .filter(e => filterType === 'all' || e.event_type === filterType)
    .filter(e => filterStatus === 'all' || e.status === filterStatus)
    .slice(0, maxEvents);

  const groupedEvents = groupEventsByDate(filteredEvents);

  const toggleExpanded = (eventId: string) => {
    setExpandedEvents(prev => {
      const next = new Set(prev);
      if (next.has(eventId)) {
        next.delete(eventId);
      } else {
        next.add(eventId);
      }
      return next;
    });
  };

  useEffect(() => {
    if (autoScroll && events.length > 0) {
      // Auto-scroll to latest event
      const timelineContainer = document.getElementById('optimization-timeline');
      if (timelineContainer) {
        timelineContainer.scrollTop = 0;
      }
    }
  }, [events.length, autoScroll]);

  return (
    <div style={{ fontFamily: 'system-ui, -apple-system, sans-serif' }}>
      {/* Header */}
      <div style={{ marginBottom: '20px' }}>
        <h2 style={{ margin: '0 0 10px 0' }}>Optimization Timeline</h2>
        <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
          <select
            value={filterType}
            onChange={e => setFilterType(e.target.value)}
            style={{ padding: '6px 12px', borderRadius: '4px', border: '1px solid #ccc' }}
          >
            <option value="all">All Types</option>
            <option value="gnn_decision">GNN Decision</option>
            <option value="council_debate">Council Debate</option>
            <option value="hitl_approval">HITL Approval</option>
            <option value="execution">Execution</option>
            <option value="rollback">Rollback</option>
          </select>

          <select
            value={filterStatus}
            onChange={e => setFilterStatus(e.target.value)}
            style={{ padding: '6px 12px', borderRadius: '4px', border: '1px solid #ccc' }}
          >
            <option value="all">All Statuses</option>
            <option value="pending">Pending</option>
            <option value="approved">Approved</option>
            <option value="rejected">Rejected</option>
            <option value="executed">Executed</option>
            <option value="rolled_back">Rolled Back</option>
          </select>

          <div style={{ marginLeft: 'auto', color: '#666', fontSize: '14px', alignSelf: 'center' }}>
            Showing {filteredEvents.length} of {events.length} events
          </div>
        </div>
      </div>

      {/* Timeline */}
      <div
        id="optimization-timeline"
        style={{
          maxHeight: '600px',
          overflowY: 'auto',
          border: '1px solid #e0e0e0',
          borderRadius: '8px',
          padding: '20px',
          background: '#fafafa'
        }}
      >
        {groupedEvents.length === 0 ? (
          <div style={{ textAlign: 'center', color: '#999', padding: '40px' }}>
            No optimization events to display
          </div>
        ) : (
          groupedEvents.map(group => (
            <div key={group.date} style={{ marginBottom: '30px' }}>
              {/* Date Header */}
              <div
                style={{
                  fontSize: '14px',
                  fontWeight: 'bold',
                  color: '#666',
                  marginBottom: '15px',
                  paddingBottom: '5px',
                  borderBottom: '2px solid #ddd'
                }}
              >
                {group.date}
              </div>

              {/* Events for this date */}
              {group.events.map(event => {
                const isExpanded = expandedEvents.has(event.id);
                const eventColor = getEventColor(event.event_type);
                const statusColor = getStatusBadgeColor(event.status);

                return (
                  <div
                    key={event.id}
                    style={{
                      marginBottom: '15px',
                      background: '#fff',
                      borderLeft: `4px solid ${eventColor}`,
                      borderRadius: '4px',
                      boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                      cursor: 'pointer',
                      transition: 'transform 0.2s',
                    }}
                    onClick={() => {
                      toggleExpanded(event.id);
                      if (onEventClick) {
                        onEventClick(event);
                      }
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.transform = 'translateX(4px)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.transform = 'translateX(0)';
                    }}
                  >
                    {/* Event Header */}
                    <div style={{ padding: '12px 16px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                          <span
                            style={{
                              fontSize: '12px',
                              fontWeight: 'bold',
                              color: '#666',
                              fontFamily: 'monospace'
                            }}
                          >
                            {formatTimestamp(event.timestamp)}
                          </span>
                          <span
                            style={{
                              fontSize: '12px',
                              padding: '2px 8px',
                              borderRadius: '3px',
                              background: eventColor,
                              color: '#fff',
                              fontWeight: 'bold'
                            }}
                          >
                            {event.event_type.replace(/_/g, ' ').toUpperCase()}
                          </span>
                          <span
                            style={{
                              fontSize: '11px',
                              padding: '2px 6px',
                              borderRadius: '3px',
                              background: statusColor,
                              color: '#fff'
                            }}
                          >
                            {event.status}
                          </span>
                        </div>
                        <div style={{ fontSize: '12px', color: '#999' }}>
                          Confidence: {(event.confidence * 100).toFixed(0)}%
                        </div>
                      </div>

                      <div style={{ marginTop: '8px', fontSize: '14px', color: '#333' }}>
                        {event.reasoning}
                      </div>

                      <div style={{ marginTop: '6px', fontSize: '12px', color: '#666' }}>
                        Cells: {event.cell_ids.join(', ')}
                      </div>
                    </div>

                    {/* Expanded Details */}
                    {isExpanded && (
                      <div
                        style={{
                          padding: '12px 16px',
                          borderTop: '1px solid #e0e0e0',
                          background: '#f9f9f9'
                        }}
                      >
                        <div style={{ fontSize: '13px', fontWeight: 'bold', marginBottom: '8px' }}>
                          Parameter Changes:
                        </div>
                        {event.parameters_changed.length === 0 ? (
                          <div style={{ fontSize: '12px', color: '#999', fontStyle: 'italic' }}>
                            No parameter changes
                          </div>
                        ) : (
                          <table style={{ width: '100%', fontSize: '12px', borderCollapse: 'collapse' }}>
                            <thead>
                              <tr style={{ background: '#eee' }}>
                                <th style={{ padding: '6px', textAlign: 'left' }}>Cell</th>
                                <th style={{ padding: '6px', textAlign: 'left' }}>Parameter</th>
                                <th style={{ padding: '6px', textAlign: 'right' }}>Old Value</th>
                                <th style={{ padding: '6px', textAlign: 'right' }}>New Value</th>
                                <th style={{ padding: '6px', textAlign: 'right' }}>Delta</th>
                              </tr>
                            </thead>
                            <tbody>
                              {event.parameters_changed.map((change, idx) => {
                                const delta = change.new_value - change.old_value;
                                return (
                                  <tr key={idx} style={{ borderBottom: '1px solid #e0e0e0' }}>
                                    <td style={{ padding: '6px' }}>{change.cell_id}</td>
                                    <td style={{ padding: '6px' }}>{change.parameter}</td>
                                    <td style={{ padding: '6px', textAlign: 'right' }}>
                                      {change.old_value.toFixed(2)}
                                    </td>
                                    <td style={{ padding: '6px', textAlign: 'right' }}>
                                      {change.new_value.toFixed(2)}
                                    </td>
                                    <td
                                      style={{
                                        padding: '6px',
                                        textAlign: 'right',
                                        color: delta > 0 ? '#2ECC71' : delta < 0 ? '#E74C3C' : '#999'
                                      }}
                                    >
                                      {delta > 0 ? '+' : ''}
                                      {delta.toFixed(2)}
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        )}

                        {event.agent_id && (
                          <div style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>
                            <strong>Agent:</strong> {event.agent_id}
                          </div>
                        )}

                        {event.kpi_impact && (
                          <div style={{ marginTop: '10px' }}>
                            <div style={{ fontSize: '13px', fontWeight: 'bold', marginBottom: '6px' }}>
                              KPI Impact:
                            </div>
                            <div style={{ fontSize: '12px', color: '#666', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '8px' }}>
                              {Object.entries(event.kpi_impact.delta_percent).map(([key, value]) => (
                                <div key={key}>
                                  <strong>{key}:</strong>{' '}
                                  <span style={{ color: value > 0 ? '#2ECC71' : value < 0 ? '#E74C3C' : '#999' }}>
                                    {value > 0 ? '+' : ''}{value.toFixed(1)}%
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          ))
        )}
      </div>
    </div>
  );
};

// ============================================================================
// Export
// ============================================================================

export default OptimizationTimeline;
