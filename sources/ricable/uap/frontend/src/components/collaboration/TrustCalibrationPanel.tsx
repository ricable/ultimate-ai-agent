// File: frontend/src/components/collaboration/TrustCalibrationPanel.tsx
import React, { useState } from 'react';
import { Card, CardHeader, CardContent } from '../ui/Card';

interface TrustCalibrationPanelProps {
  trustStatus: any;
  onTrustFeedback: (feedbackType: string, rating: number) => void;
  onRequestExplanation: () => void;
}

export function TrustCalibrationPanel({ trustStatus, onTrustFeedback, onRequestExplanation }: TrustCalibrationPanelProps) {
  const [feedbackMode, setFeedbackMode] = useState<string | null>(null);
  const [feedbackRating, setFeedbackRating] = useState(5);

  if (!trustStatus) {
    return (
      <Card>
        <CardHeader>
          <h3 className="text-lg font-semibold">Trust Calibration</h3>
        </CardHeader>
        <CardContent>
          <div className="text-center py-4">
            <div className="animate-pulse bg-gray-200 h-4 w-3/4 mx-auto mb-2 rounded"></div>
            <div className="animate-pulse bg-gray-200 h-4 w-1/2 mx-auto rounded"></div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const getTrustLevelColor = (level: string) => {
    switch (level) {
      case 'very_high': return 'text-green-700 bg-green-100';
      case 'high': return 'text-green-600 bg-green-50';
      case 'moderate': return 'text-yellow-600 bg-yellow-50';
      case 'low': return 'text-orange-600 bg-orange-50';
      case 'very_low': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getTrustIcon = (level: string) => {
    switch (level) {
      case 'very_high': return '🌟';
      case 'high': return '✅';
      case 'moderate': return '⚖️';
      case 'low': return '⚠️';
      case 'very_low': return '❌';
      default: return '❓';
    }
  };

  const handleFeedback = (type: string) => {
    setFeedbackMode(type);
    setFeedbackRating(5);
  };

  const submitFeedback = () => {
    if (feedbackMode) {
      onTrustFeedback(feedbackMode, feedbackRating);
      setFeedbackMode(null);
    }
  };

  const dimensionLabels = {
    reliability: 'Reliability',
    competence: 'Competence',
    transparency: 'Transparency',
    benevolence: 'Helpfulness',
    predictability: 'Predictability',
    integrity: 'Integrity'
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold">Trust Calibration</h3>
          <button
            onClick={onRequestExplanation}
            className="text-xs bg-gray-500 text-white px-2 py-1 rounded hover:bg-gray-600"
          >
            Explain Trust
          </button>
        </div>
      </CardHeader>
      <CardContent>
        {feedbackMode ? (
          <div className="space-y-4">
            <div className="bg-blue-50 p-4 rounded-lg">
              <h4 className="font-medium text-blue-900 mb-2">
                Rate my {feedbackMode.replace('_', ' ')}
              </h4>
              <div className="flex items-center space-x-2 mb-3">
                {[1, 2, 3, 4, 5].map(rating => (
                  <button
                    key={rating}
                    onClick={() => setFeedbackRating(rating)}
                    className={`w-8 h-8 rounded-full border-2 flex items-center justify-center text-sm font-medium ${
                      rating <= feedbackRating
                        ? 'bg-blue-500 border-blue-500 text-white'
                        : 'bg-white border-gray-300 text-gray-500 hover:border-blue-300'
                    }`}
                  >
                    {rating}
                  </button>
                ))}
              </div>
              <div className="flex space-x-2">
                <button
                  onClick={submitFeedback}
                  className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
                >
                  Submit
                </button>
                <button
                  onClick={() => setFeedbackMode(null)}
                  className="bg-gray-300 text-gray-700 px-4 py-2 rounded hover:bg-gray-400"
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Overall Trust */}
            <div className="text-center">
              <div className="text-2xl mb-2">
                {getTrustIcon(trustStatus.trust_level)}
              </div>
              <div className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${
                getTrustLevelColor(trustStatus.trust_level)
              }`}>
                {trustStatus.trust_level.replace('_', ' ').toUpperCase()}
              </div>
              <div className="text-2xl font-bold text-gray-700 mt-2">
                {(trustStatus.overall_trust * 100).toFixed(0)}%
              </div>
              <div className="text-xs text-gray-500">Overall Trust</div>
            </div>

            {/* Trust Dimensions */}
            <div className="space-y-3">
              <h4 className="text-sm font-medium text-gray-700">Trust Dimensions</h4>
              {Object.entries(trustStatus.dimension_trust || {}).map(([dimension, score]) => (
                <div key={dimension} className="flex items-center justify-between">
                  <span className="text-xs text-gray-600 capitalize">
                    {dimensionLabels[dimension as keyof typeof dimensionLabels] || dimension}
                  </span>
                  <div className="flex items-center space-x-2">
                    <div className="w-16 bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{ width: `${(score as number) * 100}%` }}
                      ></div>
                    </div>
                    <span className="text-xs text-gray-500 w-8">
                      {((score as number) * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>

            {/* Trust Trends */}
            {trustStatus.trust_trends && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-gray-700">Recent Trends</h4>
                {Object.entries(trustStatus.trust_trends).map(([dimension, trend]) => (
                  <div key={dimension} className="flex items-center justify-between text-xs">
                    <span className="text-gray-600 capitalize">
                      {dimensionLabels[dimension as keyof typeof dimensionLabels] || dimension}
                    </span>
                    <span className={`font-medium ${
                      (trend as number) > 0.01 ? 'text-green-600' :
                      (trend as number) < -0.01 ? 'text-red-600' : 'text-gray-500'
                    }`}>
                      {(trend as number) > 0.01 ? '↗️' :
                       (trend as number) < -0.01 ? '↘️' : '→'}
                      {((trend as number) * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            )}

            {/* Recent Events */}
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-700">Trust Metrics</h4>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-gray-50 p-2 rounded">
                  <div className="font-medium">Calibrations</div>
                  <div className="text-blue-600">{trustStatus.calibration_count || 0}</div>
                </div>
                <div className="bg-gray-50 p-2 rounded">
                  <div className="font-medium">Events (1h)</div>
                  <div className="text-green-600">{trustStatus.recent_events_count || 0}</div>
                </div>
              </div>
            </div>

            {/* Recommendations */}
            {trustStatus.recommendations && trustStatus.recommendations.length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-gray-700">Recommendations</h4>
                <div className="space-y-1">
                  {trustStatus.recommendations.slice(0, 3).map((rec: string, index: number) => (
                    <div key={index} className="text-xs bg-yellow-50 text-yellow-800 p-2 rounded border-l-2 border-yellow-300">
                      {rec.replace(/_/g, ' ').toLowerCase()}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Feedback Buttons */}
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => handleFeedback('helpful_suggestion')}
                className="bg-green-500 text-white text-xs py-2 px-3 rounded hover:bg-green-600"
              >
                👍 Helpful
              </button>
              <button
                onClick={() => handleFeedback('clear_explanation')}
                className="bg-blue-500 text-white text-xs py-2 px-3 rounded hover:bg-blue-600"
              >
                💡 Clear
              </button>
              <button
                onClick={() => handleFeedback('correct_prediction')}
                className="bg-purple-500 text-white text-xs py-2 px-3 rounded hover:bg-purple-600"
              >
                ✅ Accurate
              </button>
              <button
                onClick={() => handleFeedback('user_feedback_positive')}
                className="bg-orange-500 text-white text-xs py-2 px-3 rounded hover:bg-orange-600"
              >
                ⭐ Overall Good
              </button>
            </div>

            {/* Trust History Indicator */}
            <div className="text-center pt-2 border-t">
              <div className="text-xs text-gray-500">
                Last updated: {new Date(trustStatus.last_calibration).toLocaleTimeString()}
              </div>
              <div className="text-xs text-gray-400">
                Baseline: {(trustStatus.trust_baseline * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}