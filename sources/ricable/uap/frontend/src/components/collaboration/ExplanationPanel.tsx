// File: frontend/src/components/collaboration/ExplanationPanel.tsx
import React, { useState } from 'react';
import { Card, CardHeader, CardContent } from '../ui/Card';

interface ExplanationPanelProps {
  explanations: any[];
  onRequestExplanation: (target: string, type?: string) => void;
  onExplanationFeedback: (explanationId: string, feedback: any) => void;
}

export function ExplanationPanel({ 
  explanations, 
  onRequestExplanation, 
  onExplanationFeedback 
}: ExplanationPanelProps) {
  const [expandedExplanation, setExpandedExplanation] = useState<string | null>(null);
  const [feedbackMode, setFeedbackMode] = useState<string | null>(null);
  const [feedbackForm, setFeedbackForm] = useState({
    clarity_rating: 5,
    helpfulness_rating: 5,
    completeness_rating: 5,
    satisfaction_rating: 5,
    feedback_text: '',
    improvement_suggestions: [] as string[]
  });
  const [explanationRequest, setExplanationRequest] = useState({
    target: '',
    type: 'causal'
  });

  const getExplanationTypeIcon = (type: string) => {
    switch (type) {
      case 'causal': return '🔍';
      case 'procedural': return '📋';
      case 'contextual': return '🌍';
      case 'comparative': return '⚖️';
      case 'predictive': return '🔮';
      case 'counterfactual': return '❓';
      case 'example_based': return '📚';
      case 'analogical': return '🔄';
      default: return '💡';
    }
  };

  const getExplanationStyleColor = (style: string) => {
    switch (style) {
      case 'concise': return 'bg-blue-100 text-blue-800';
      case 'detailed': return 'bg-purple-100 text-purple-800';
      case 'step_by_step': return 'bg-green-100 text-green-800';
      case 'visual': return 'bg-orange-100 text-orange-800';
      case 'narrative': return 'bg-pink-100 text-pink-800';
      case 'analytical': return 'bg-gray-100 text-gray-800';
      case 'intuitive': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const handleRequestExplanation = () => {
    if (explanationRequest.target.trim()) {
      onRequestExplanation(explanationRequest.target, explanationRequest.type);
      setExplanationRequest({ target: '', type: 'causal' });
    }
  };

  const handleFeedbackSubmit = (explanationId: string) => {
    onExplanationFeedback(explanationId, feedbackForm);
    setFeedbackMode(null);
    setFeedbackForm({
      clarity_rating: 5,
      helpfulness_rating: 5,
      completeness_rating: 5,
      satisfaction_rating: 5,
      feedback_text: '',
      improvement_suggestions: []
    });
  };

  const startFeedback = (explanationId: string) => {
    setFeedbackMode(explanationId);
  };

  const toggleExpansion = (explanationId: string) => {
    setExpandedExplanation(
      expandedExplanation === explanationId ? null : explanationId
    );
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMinutes = Math.round((now.getTime() - date.getTime()) / 60000);
    
    if (diffMinutes < 1) return 'just now';
    if (diffMinutes < 60) return `${diffMinutes}m ago`;
    if (diffMinutes < 1440) return `${Math.round(diffMinutes / 60)}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold">AI Explanations</h3>
          <span className="text-xs text-gray-500">
            {explanations.length} explanations
          </span>
        </div>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          
          {/* Request New Explanation */}
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-gray-700">Request Explanation</h4>
            <div className="space-y-2">
              <input
                type="text"
                value={explanationRequest.target}
                onChange={(e) => setExplanationRequest({
                  ...explanationRequest,
                  target: e.target.value
                })}
                placeholder="What would you like me to explain?"
                className="w-full text-sm border border-gray-300 rounded p-2"
              />
              
              <div className="flex space-x-2">
                <select
                  value={explanationRequest.type}
                  onChange={(e) => setExplanationRequest({
                    ...explanationRequest,
                    type: e.target.value
                  })}
                  className="text-sm border border-gray-300 rounded p-2"
                >
                  <option value="causal">Why (Causal)</option>
                  <option value="procedural">How (Procedural)</option>
                  <option value="contextual">Context</option>
                  <option value="comparative">Compare</option>
                  <option value="predictive">Predict</option>
                  <option value="example_based">Examples</option>
                  <option value="analogical">Analogy</option>
                </select>
                
                <button
                  onClick={handleRequestExplanation}
                  disabled={!explanationRequest.target.trim()}
                  className="bg-blue-500 text-white px-4 py-2 rounded text-sm hover:bg-blue-600 disabled:bg-gray-400"
                >
                  Explain
                </button>
              </div>
            </div>
          </div>

          {/* Quick Explanation Buttons */}
          <div className="space-y-2">
            <h4 className="text-sm font-medium text-gray-700">Quick Explanations</h4>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => onRequestExplanation('current decision', 'causal')}
                className="text-xs bg-blue-100 text-blue-800 p-2 rounded hover:bg-blue-200"
              >
                🔍 Why this decision?
              </button>
              <button
                onClick={() => onRequestExplanation('system behavior', 'procedural')}
                className="text-xs bg-green-100 text-green-800 p-2 rounded hover:bg-green-200"
              >
                📋 How does this work?
              </button>
              <button
                onClick={() => onRequestExplanation('recommendations', 'comparative')}
                className="text-xs bg-purple-100 text-purple-800 p-2 rounded hover:bg-purple-200"
              >
                ⚖️ Compare options
              </button>
              <button
                onClick={() => onRequestExplanation('next steps', 'predictive')}
                className="text-xs bg-orange-100 text-orange-800 p-2 rounded hover:bg-orange-200"
              >
                🔮 What happens next?
              </button>
            </div>
          </div>

          {/* Explanations List */}
          {explanations.length > 0 ? (
            <div className="space-y-3">
              <h4 className="text-sm font-medium text-gray-700">
                Recent Explanations ({explanations.length})
              </h4>
              
              {explanations.slice(0, 5).map((explanation) => (
                <div key={explanation.explanation_id} className="border rounded-lg p-3 bg-gray-50">
                  
                  {/* Explanation Header */}
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className="text-lg">
                        {getExplanationTypeIcon(explanation.explanation_type)}
                      </span>
                      <span className="text-sm font-medium capitalize">
                        {explanation.explanation_type.replace('_', ' ')} explanation
                      </span>
                      <span className={`text-xs px-2 py-1 rounded ${
                        getExplanationStyleColor(explanation.explanation_style)
                      }`}>
                        {explanation.explanation_style}
                      </span>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <span className={`text-xs font-medium ${
                        getConfidenceColor(explanation.confidence)
                      }`}>
                        {(explanation.confidence * 100).toFixed(0)}%
                      </span>
                      <button
                        onClick={() => toggleExpansion(explanation.explanation_id)}
                        className="text-xs text-gray-500 hover:text-gray-700"
                      >
                        {expandedExplanation === explanation.explanation_id ? '−' : '+'}
                      </button>
                    </div>
                  </div>

                  {/* Explanation Content */}
                  <div className="text-sm text-gray-700 mb-2">
                    {explanation.content.length > 200 && expandedExplanation !== explanation.explanation_id
                      ? explanation.content.substring(0, 200) + '...'
                      : explanation.content
                    }
                  </div>

                  {/* Quality Metrics */}
                  <div className="flex items-center space-x-4 text-xs text-gray-500 mb-2">
                    <span>Clarity: {(explanation.clarity_score * 100).toFixed(0)}%</span>
                    <span>Complete: {(explanation.completeness * 100).toFixed(0)}%</span>
                    <span>Time: {explanation.generation_time_ms.toFixed(0)}ms</span>
                    <span>{formatTimestamp(explanation.generated_at)}</span>
                  </div>

                  {/* Expanded Details */}
                  {expandedExplanation === explanation.explanation_id && (
                    <div className="space-y-3 mt-3 pt-3 border-t">
                      
                      {/* Supporting Evidence */}
                      {explanation.supporting_evidence && explanation.supporting_evidence.length > 0 && (
                        <div>
                          <h5 className="text-xs font-medium text-gray-700 mb-1">
                            📊 Supporting Evidence
                          </h5>
                          <ul className="text-xs text-gray-600 space-y-1">
                            {explanation.supporting_evidence.map((evidence: string, index: number) => (
                              <li key={index} className="flex items-start space-x-1">
                                <span>•</span>
                                <span>{evidence}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Key Concepts */}
                      {explanation.key_concepts && explanation.key_concepts.length > 0 && (
                        <div>
                          <h5 className="text-xs font-medium text-gray-700 mb-1">
                            🔑 Key Concepts
                          </h5>
                          <div className="flex flex-wrap gap-1">
                            {explanation.key_concepts.map((concept: string, index: number) => (
                              <span
                                key={index}
                                className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded"
                              >
                                {concept}
                              </span>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Assumptions */}
                      {explanation.assumptions && explanation.assumptions.length > 0 && (
                        <div>
                          <h5 className="text-xs font-medium text-gray-700 mb-1">
                            ⚠️ Assumptions
                          </h5>
                          <ul className="text-xs text-gray-600 space-y-1">
                            {explanation.assumptions.map((assumption: string, index: number) => (
                              <li key={index} className="flex items-start space-x-1">
                                <span>•</span>
                                <span>{assumption}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Limitations */}
                      {explanation.limitations && explanation.limitations.length > 0 && (
                        <div>
                          <h5 className="text-xs font-medium text-gray-700 mb-1">
                            ⚠️ Limitations
                          </h5>
                          <ul className="text-xs text-gray-600 space-y-1">
                            {explanation.limitations.map((limitation: string, index: number) => (
                              <li key={index} className="flex items-start space-x-1">
                                <span>•</span>
                                <span>{limitation}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Follow-up Questions */}
                      {explanation.follow_up_questions && explanation.follow_up_questions.length > 0 && (
                        <div>
                          <h5 className="text-xs font-medium text-gray-700 mb-1">
                            ❓ Follow-up Questions
                          </h5>
                          <div className="space-y-1">
                            {explanation.follow_up_questions.map((question: string, index: number) => (
                              <button
                                key={index}
                                onClick={() => onRequestExplanation(question, 'causal')}
                                className="block w-full text-left text-xs text-blue-600 hover:text-blue-800 hover:bg-blue-50 p-1 rounded"
                              >
                                {question}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Feedback Section */}
                  {feedbackMode === explanation.explanation_id ? (
                    <div className="mt-3 pt-3 border-t space-y-3">
                      <h5 className="text-sm font-medium text-gray-700">Rate this explanation</h5>
                      
                      {/* Rating Sliders */}
                      <div className="space-y-2">
                        {[
                          { key: 'clarity_rating', label: 'Clarity' },
                          { key: 'helpfulness_rating', label: 'Helpfulness' },
                          { key: 'completeness_rating', label: 'Completeness' },
                          { key: 'satisfaction_rating', label: 'Overall Satisfaction' }
                        ].map(({ key, label }) => (
                          <div key={key} className="flex items-center space-x-2">
                            <span className="text-xs text-gray-600 w-20">{label}:</span>
                            <input
                              type="range"
                              min="1"
                              max="5"
                              value={feedbackForm[key as keyof typeof feedbackForm] as number}
                              onChange={(e) => setFeedbackForm({
                                ...feedbackForm,
                                [key]: parseInt(e.target.value)
                              })}
                              className="flex-1"
                            />
                            <span className="text-xs text-gray-500 w-8">
                              {feedbackForm[key as keyof typeof feedbackForm]}
                            </span>
                          </div>
                        ))}
                      </div>

                      {/* Feedback Text */}
                      <textarea
                        value={feedbackForm.feedback_text}
                        onChange={(e) => setFeedbackForm({
                          ...feedbackForm,
                          feedback_text: e.target.value
                        })}
                        placeholder="Additional feedback (optional)..."
                        className="w-full text-xs border border-gray-300 rounded p-2 h-16 resize-none"
                      />

                      {/* Buttons */}
                      <div className="flex space-x-2">
                        <button
                          onClick={() => handleFeedbackSubmit(explanation.explanation_id)}
                          className="bg-green-500 text-white px-3 py-1 rounded text-xs hover:bg-green-600"
                        >
                          Submit Feedback
                        </button>
                        <button
                          onClick={() => setFeedbackMode(null)}
                          className="bg-gray-300 text-gray-700 px-3 py-1 rounded text-xs hover:bg-gray-400"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className="flex justify-between items-center mt-2">
                      <button
                        onClick={() => startFeedback(explanation.explanation_id)}
                        className="text-xs bg-gray-200 text-gray-700 px-2 py-1 rounded hover:bg-gray-300"
                      >
                        📝 Rate
                      </button>
                      
                      <div className="flex space-x-2">
                        <button
                          onClick={() => onRequestExplanation(
                            `more details about: ${explanation.content.substring(0, 50)}`,
                            'detailed'
                          )}
                          className="text-xs text-blue-600 hover:text-blue-800"
                        >
                          More details
                        </button>
                        <button
                          onClick={() => onRequestExplanation(
                            `alternative explanation for: ${explanation.content.substring(0, 50)}`,
                            'comparative'
                          )}
                          className="text-xs text-purple-600 hover:text-purple-800"
                        >
                          Alternative
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              ))}

              {explanations.length > 5 && (
                <div className="text-center">
                  <button className="text-xs text-gray-500 hover:text-gray-700">
                    Show {explanations.length - 5} more explanations...
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-8">
              <div className="text-4xl mb-2">💡</div>
              <p className="text-sm text-gray-600 mb-2">No explanations yet</p>
              <p className="text-xs text-gray-500">
                Request an explanation to get AI insights and reasoning
              </p>
            </div>
          )}

          {/* Explanation Statistics */}
          {explanations.length > 0 && (
            <div className="border-t pt-3">
              <h4 className="text-sm font-medium text-gray-700 mb-2">📊 Explanation Stats</h4>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-gray-50 p-2 rounded">
                  <div className="font-medium">Average Clarity</div>
                  <div className="text-blue-600">
                    {explanations.length > 0 
                      ? Math.round(explanations.reduce((sum, exp) => sum + exp.clarity_score, 0) / explanations.length * 100)
                      : 0}%
                  </div>
                </div>
                <div className="bg-gray-50 p-2 rounded">
                  <div className="font-medium">Average Confidence</div>
                  <div className="text-green-600">
                    {explanations.length > 0
                      ? Math.round(explanations.reduce((sum, exp) => sum + exp.confidence, 0) / explanations.length * 100)
                      : 0}%
                  </div>
                </div>
                <div className="bg-gray-50 p-2 rounded">
                  <div className="font-medium">Most Used Type</div>
                  <div className="text-purple-600">
                    {explanations.length > 0
                      ? (() => {
                          const counts = explanations.reduce((acc, curr) => {
                            acc[curr.explanation_type] = (acc[curr.explanation_type] || 0) + 1;
                            return acc;
                          }, {} as any);
                          const mostUsed = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
                          return mostUsed.replace('_', ' ');
                        })()
                      : 'None'
                    }
                  </div>
                </div>
                <div className="bg-gray-50 p-2 rounded">
                  <div className="font-medium">Avg Gen Time</div>
                  <div className="text-orange-600">
                    {explanations.length > 0
                      ? Math.round(explanations.reduce((sum, exp) => sum + exp.generation_time_ms, 0) / explanations.length)
                      : 0}ms
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}