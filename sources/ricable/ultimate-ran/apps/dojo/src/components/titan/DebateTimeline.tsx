/**
 * DebateTimeline - Council Debate Visualization
 * Part of Ericsson Gen 7.0 Neuro-Symbolic Titan Platform
 *
 * Displays THINKING_STEP events as a vertical timeline.
 * Shows real-time debate flow between council members.
 */

'use client';

import React, { useRef, useEffect } from 'react';
import type { ThinkingStepEvent, CouncilMember } from '../../types/council';

interface DebateTimelineProps {
  events: ThinkingStepEvent[];
  councilMembers: CouncilMember[];
  isDebating: boolean;
  className?: string;
}

const statusColors = {
  THINKING: 'bg-blue-500/20 border-blue-500 text-blue-300',
  CRITIQUE: 'bg-yellow-500/20 border-yellow-500 text-yellow-300',
  PROPOSAL: 'bg-green-500/20 border-green-500 text-green-300',
  CONSENSUS: 'bg-purple-500/20 border-purple-500 text-purple-300',
  DENIED: 'bg-red-500/20 border-red-500 text-red-300'
};

const statusIcons = {
  THINKING: '💭',
  CRITIQUE: '🔍',
  PROPOSAL: '💡',
  CONSENSUS: '✅',
  DENIED: '❌'
};

export function DebateTimeline({
  events,
  councilMembers,
  isDebating,
  className = ''
}: DebateTimelineProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to latest event
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events]);

  const getMemberInfo = (agentId: string) => {
    return councilMembers.find(m => m.id === agentId);
  };

  return (
    <div className={`relative ${className}`}>
      {/* Glass Box Container */}
      <div className="bg-gray-900/50 backdrop-blur-sm rounded-lg border border-gray-700 p-4 h-full flex flex-col">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <span className="text-2xl">💬</span>
            Council Debate Stream
          </h3>
          {isDebating && (
            <div className="flex items-center gap-2 text-sm text-blue-400">
              <div className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
              Debating...
            </div>
          )}
        </div>

        {/* Timeline Container */}
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto space-y-3 pr-2 custom-scrollbar"
          style={{ maxHeight: '600px' }}
        >
          {events.length === 0 ? (
            <div className="flex items-center justify-center h-full text-gray-400">
              <p>Waiting for council debate...</p>
            </div>
          ) : (
            events.map((event, index) => {
              const member = getMemberInfo(event.agentId);
              const isLastEvent = index === events.length - 1;

              return (
                <div
                  key={`${event.agentId}-${event.timestamp}-${index}`}
                  className={`relative pl-8 ${isLastEvent ? 'animate-fadeIn' : ''}`}
                >
                  {/* Timeline Line */}
                  {index < events.length - 1 && (
                    <div className="absolute left-4 top-8 bottom-0 w-px bg-gray-700" />
                  )}

                  {/* Timeline Node */}
                  <div className="absolute left-0 top-2 w-8 h-8 rounded-full bg-gray-800 border-2 border-gray-700 flex items-center justify-center">
                    <span className="text-lg">{member?.avatar || '🤖'}</span>
                  </div>

                  {/* Event Card */}
                  <div className={`rounded-lg border p-3 ${statusColors[event.status]}`}>
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-white">{event.agentName}</span>
                        <span className="text-xs px-2 py-1 rounded bg-gray-900/50">
                          {event.role}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-xs">
                        <span>{statusIcons[event.status]}</span>
                        <span className="opacity-75">{event.status}</span>
                      </div>
                    </div>

                    <p className="text-sm text-gray-100 leading-relaxed">
                      {event.content}
                    </p>

                    {event.metadata && Object.keys(event.metadata).length > 0 && (
                      <div className="mt-2 pt-2 border-t border-gray-700/50">
                        <details className="text-xs">
                          <summary className="cursor-pointer text-gray-400 hover:text-gray-300">
                            View metadata
                          </summary>
                          <pre className="mt-2 p-2 bg-gray-900/50 rounded overflow-x-auto">
                            {JSON.stringify(event.metadata, null, 2)}
                          </pre>
                        </details>
                      </div>
                    )}

                    <div className="mt-2 text-xs text-gray-400">
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Council Members Summary */}
        <div className="mt-4 pt-4 border-t border-gray-700">
          <div className="flex items-center gap-3 text-sm">
            <span className="text-gray-400">Active Council:</span>
            {councilMembers.map(member => (
              <div
                key={member.id}
                className="flex items-center gap-1 px-2 py-1 bg-gray-800 rounded"
                title={member.model_id}
              >
                <span>{member.avatar}</span>
                <span className="text-gray-300 capitalize">{member.role}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(0, 0, 0, 0.2);
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(100, 100, 100, 0.5);
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(150, 150, 150, 0.7);
        }
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(-10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        .animate-fadeIn {
          animation: fadeIn 0.3s ease-out;
        }
      `}</style>
    </div>
  );
}
