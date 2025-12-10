// File: frontend/src/components/collaboration/UserCollaborationProfile.tsx
import React, { useState } from 'react';
import { Card, CardHeader, CardContent } from '../ui/Card';

interface UserCollaborationProfileProps {
  userProfile: any;
  currentSession: any;
  onProfileUpdate: () => void;
}

export function UserCollaborationProfile({ userProfile, currentSession, onProfileUpdate }: UserCollaborationProfileProps) {
  const [isEditingProfile, setIsEditingProfile] = useState(false);
  const [editableProfile, setEditableProfile] = useState<any>(null);

  if (!userProfile) {
    return (
      <Card>
        <CardHeader>
          <h3 className="text-lg font-semibold">User Profile</h3>
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

  const handleEditProfile = () => {
    setEditableProfile({ ...userProfile });
    setIsEditingProfile(true);
  };

  const handleSaveProfile = async () => {
    try {
      const response = await fetch(`http://localhost:8000/api/collaboration/update-profile`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(editableProfile)
      });

      if (response.ok) {
        onProfileUpdate();
        setIsEditingProfile(false);
      }
    } catch (err) {
      console.error('Failed to update profile:', err);
    }
  };

  const getCollaborationModeColor = (mode: string) => {
    switch (mode) {
      case 'assistant': return 'bg-blue-100 text-blue-800';
      case 'cooperative': return 'bg-green-100 text-green-800';
      case 'autonomous': return 'bg-purple-100 text-purple-800';
      case 'advisory': return 'bg-yellow-100 text-yellow-800';
      case 'supervisory': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getCognitiveStyleIcon = (style: string) => {
    switch (style) {
      case 'analytical': return '📊';
      case 'intuitive': return '💡';
      case 'visual': return '👁️';
      case 'mixed': return '🔄';
      default: return '🤔';
    }
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex justify-between items-center">
          <h3 className="text-lg font-semibold">Collaboration Profile</h3>
          {!isEditingProfile && (
            <button
              onClick={handleEditProfile}
              className="text-sm bg-blue-500 text-white px-3 py-1 rounded hover:bg-blue-600"
            >
              Edit
            </button>
          )}
        </div>
      </CardHeader>
      <CardContent>
        {isEditingProfile ? (
          <div className="space-y-4">
            {/* Editable Profile Form */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Expertise Level
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={editableProfile.expertise_level}
                onChange={(e) => setEditableProfile({
                  ...editableProfile,
                  expertise_level: parseFloat(e.target.value)
                })}
                className="w-full"
              />
              <span className="text-xs text-gray-500">
                {(editableProfile.expertise_level * 100).toFixed(0)}%
              </span>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Preferred Collaboration Mode
              </label>
              <select
                value={editableProfile.preferred_collaboration_mode}
                onChange={(e) => setEditableProfile({
                  ...editableProfile,
                  preferred_collaboration_mode: e.target.value
                })}
                className="w-full border border-gray-300 rounded px-3 py-2"
              >
                <option value="assistant">Assistant</option>
                <option value="cooperative">Cooperative</option>
                <option value="autonomous">Autonomous</option>
                <option value="advisory">Advisory</option>
                <option value="supervisory">Supervisory</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Cognitive Style
              </label>
              <select
                value={editableProfile.cognitive_style}
                onChange={(e) => setEditableProfile({
                  ...editableProfile,
                  cognitive_style: e.target.value
                })}
                className="w-full border border-gray-300 rounded px-3 py-2"
              >
                <option value="analytical">Analytical</option>
                <option value="intuitive">Intuitive</option>
                <option value="visual">Visual</option>
                <option value="mixed">Mixed</option>
              </select>
            </div>

            <div className="flex space-x-2">
              <button
                onClick={handleSaveProfile}
                className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
              >
                Save
              </button>
              <button
                onClick={() => setIsEditingProfile(false)}
                className="bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600"
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {/* Profile Display */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <h4 className="text-sm font-medium text-gray-700">Expertise Level</h4>
                <div className="mt-1">
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full"
                      style={{ width: `${userProfile.expertise_level * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-xs text-gray-500 mt-1">
                    {(userProfile.expertise_level * 100).toFixed(0)}% Expert
                  </span>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium text-gray-700">Trust Level</h4>
                <div className="mt-1">
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-green-600 h-2 rounded-full"
                      style={{ width: `${userProfile.trust_level * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-xs text-gray-500 mt-1">
                    {(userProfile.trust_level * 100).toFixed(0)}% Trust
                  </span>
                </div>
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium text-gray-700">Collaboration Mode</h4>
              <span className={`inline-block mt-1 px-2 py-1 rounded-full text-xs font-medium ${
                getCollaborationModeColor(userProfile.preferred_collaboration_mode)
              }`}>
                {userProfile.preferred_collaboration_mode}
              </span>
            </div>

            <div>
              <h4 className="text-sm font-medium text-gray-700">Cognitive Style</h4>
              <div className="mt-1 flex items-center space-x-2">
                <span className="text-lg">
                  {getCognitiveStyleIcon(userProfile.cognitive_style)}
                </span>
                <span className="text-sm capitalize">{userProfile.cognitive_style}</span>
              </div>
            </div>

            <div>
              <h4 className="text-sm font-medium text-gray-700">Performance Metrics</h4>
              <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                <div className="bg-gray-50 p-2 rounded">
                  <div className="font-medium">Accuracy</div>
                  <div className="text-green-600">
                    {((userProfile.performance_metrics?.recent_accuracy || 0.75) * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-gray-50 p-2 rounded">
                  <div className="font-medium">Efficiency</div>
                  <div className="text-blue-600">
                    {((userProfile.performance_metrics?.efficiency || 0.8) * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>

            {/* Session Information */}
            {currentSession && (
              <div className="border-t pt-4">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Current Session</h4>
                <div className="text-xs space-y-1">
                  <div className="flex justify-between">
                    <span>Task:</span>
                    <span className="text-gray-600">{currentSession.task_description}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Duration:</span>
                    <span className="text-gray-600">
                      {Math.round((new Date().getTime() - new Date(currentSession.started_at).getTime()) / 60000)}m
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Interactions:</span>
                    <span className="text-gray-600">{currentSession.interaction_history?.length || 0}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Adaptations:</span>
                    <span className="text-gray-600">{currentSession.adaptation_log?.length || 0}</span>
                  </div>
                </div>
              </div>
            )}

            {/* Preferences */}
            <div className="border-t pt-4">
              <h4 className="text-sm font-medium text-gray-700 mb-2">Preferences</h4>
              <div className="text-xs space-y-1">
                <div className="flex justify-between">
                  <span>Information Density:</span>
                  <span className="text-gray-600">
                    {userProfile.preferences?.information_density || 'Medium'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Explanation Style:</span>
                  <span className="text-gray-600">
                    {userProfile.preferences?.explanation_style || 'Adaptive'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Feedback Frequency:</span>
                  <span className="text-gray-600">
                    {userProfile.preferences?.feedback_frequency || 'As needed'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}