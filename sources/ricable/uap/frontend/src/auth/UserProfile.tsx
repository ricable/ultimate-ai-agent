// User profile component
import React, { useState } from 'react';
import { useAuth } from './AuthContext';
import { usePermissions } from './ProtectedRoute';

interface UserProfileProps {
  className?: string;
  showLogout?: boolean;
}

export const UserProfile: React.FC<UserProfileProps> = ({
  className = '',
  showLogout = true
}) => {
  const { user, logout, isLoading } = useAuth();
  const { userRoles, hasPermission } = usePermissions();
  const [isExpanded, setIsExpanded] = useState(false);

  if (!user) {
    return null;
  }

  const handleLogout = async () => {
    if (window.confirm('Are you sure you want to logout?')) {
      await logout();
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getRoleBadgeColor = (role: string) => {
    const colors: Record<string, string> = {
      admin: 'bg-red-100 text-red-800',
      manager: 'bg-blue-100 text-blue-800',
      user: 'bg-green-100 text-green-800',
      guest: 'bg-gray-100 text-gray-800'
    };
    return colors[role] || 'bg-gray-100 text-gray-800';
  };

  const getStatusColor = (isActive: boolean, isVerified: boolean) => {
    if (!isActive) return 'text-red-600';
    if (!isVerified) return 'text-yellow-600';
    return 'text-green-600';
  };

  const getStatusText = (isActive: boolean, isVerified: boolean) => {
    if (!isActive) return 'Inactive';
    if (!isVerified) return 'Unverified';
    return 'Active';
  };

  return (
    <div className={`bg-white rounded-lg shadow-md ${className}`}>
      {/* Compact View */}
      <div 
        className="p-4 cursor-pointer hover:bg-gray-50 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {/* Avatar */}
            <div className="w-10 h-10 bg-blue-600 text-white rounded-full flex items-center justify-center font-semibold">
              {(user.full_name || user.username).charAt(0).toUpperCase()}
            </div>
            
            {/* User Info */}
            <div>
              <h3 className="font-semibold text-gray-900">
                {user.full_name || user.username}
              </h3>
              <p className="text-sm text-gray-600">@{user.username}</p>
            </div>
          </div>

          {/* Toggle Icon */}
          <div className="text-gray-400">
            <svg
              className={`w-5 h-5 transform transition-transform ${isExpanded ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </div>
      </div>

      {/* Expanded View */}
      {isExpanded && (
        <div className="border-t border-gray-200 p-4 space-y-4">
          {/* Detailed User Information */}
          <div className="grid grid-cols-1 gap-3 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Email:</span>
              <span className="font-medium">{user.email}</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-gray-600">User ID:</span>
              <span className="font-mono text-xs">{user.id}</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-gray-600">Status:</span>
              <span className={`font-medium ${getStatusColor(user.is_active, user.is_verified)}`}>
                {getStatusText(user.is_active, user.is_verified)}
              </span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-gray-600">Created:</span>
              <span>{formatDate(user.created_at)}</span>
            </div>
            
            {user.last_login && (
              <div className="flex justify-between">
                <span className="text-gray-600">Last Login:</span>
                <span>{formatDate(user.last_login)}</span>
              </div>
            )}
          </div>

          {/* Roles */}
          <div>
            <span className="text-sm text-gray-600 block mb-2">Roles:</span>
            <div className="flex flex-wrap gap-2">
              {userRoles.map(role => (
                <span
                  key={role}
                  className={`px-2 py-1 text-xs font-medium rounded-full ${getRoleBadgeColor(role)}`}
                >
                  {role}
                </span>
              ))}
            </div>
          </div>

          {/* Permissions (for admins/managers) */}
          {hasPermission('system:read') && (
            <div>
              <span className="text-sm text-gray-600 block mb-2">Key Permissions:</span>
              <div className="grid grid-cols-2 gap-1 text-xs">
                {[
                  'agent:create',
                  'agent:read',
                  'system:read',
                  'websocket:connect'
                ].map(permission => (
                  <div key={permission} className="flex items-center space-x-1">
                    <span className={hasPermission(permission as any) ? 'text-green-600' : 'text-gray-400'}>
                      {hasPermission(permission as any) ? '✓' : '✗'}
                    </span>
                    <span className="truncate">{permission}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Metadata (if any) */}
          {Object.keys(user.metadata || {}).length > 0 && (
            <div>
              <span className="text-sm text-gray-600 block mb-2">Additional Info:</span>
              <div className="text-xs bg-gray-50 p-2 rounded">
                <pre className="whitespace-pre-wrap">
                  {JSON.stringify(user.metadata, null, 2)}
                </pre>
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-2 pt-2 border-t border-gray-100">
            {showLogout && (
              <button
                onClick={handleLogout}
                disabled={isLoading}
                className="flex-1 px-4 py-2 text-sm font-medium text-red-600 bg-red-50 hover:bg-red-100 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? 'Logging out...' : 'Logout'}
              </button>
            )}
            
            <button
              onClick={() => setIsExpanded(false)}
              className="px-4 py-2 text-sm font-medium text-gray-600 bg-gray-50 hover:bg-gray-100 rounded-md transition-colors"
            >
              Collapse
            </button>
          </div>
        </div>
      )}
    </div>
  );
};