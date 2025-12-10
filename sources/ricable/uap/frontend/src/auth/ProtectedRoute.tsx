// Protected route component that requires authentication
import React from 'react';
import { useAuth } from './AuthContext';
import { AuthPage } from './AuthPage';
import { Permission } from './types';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredPermission?: Permission;
  fallback?: React.ReactNode;
  redirectTo?: 'login' | 'register';
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requiredPermission,
  fallback,
  redirectTo = 'login'
}) => {
  const { user, isAuthenticated, isLoading } = useAuth();

  // Show loading state while checking authentication
  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  // If not authenticated, show auth page
  if (!isAuthenticated) {
    if (fallback) {
      return <>{fallback}</>;
    }
    return <AuthPage defaultMode={redirectTo} />;
  }

  // Check permission if required
  if (requiredPermission && user) {
    const hasPermission = checkUserPermission(user, requiredPermission);
    
    if (!hasPermission) {
      return (
        <div className="min-h-screen bg-gray-100 flex items-center justify-center">
          <div className="max-w-md w-full bg-white rounded-lg shadow-md p-6 text-center">
            <div className="text-red-500 text-6xl mb-4">🚫</div>
            <h2 className="text-2xl font-bold text-gray-800 mb-2">Access Denied</h2>
            <p className="text-gray-600 mb-4">
              You don't have permission to access this resource.
            </p>
            <p className="text-sm text-gray-500">
              Required permission: <code className="bg-gray-100 px-2 py-1 rounded">{requiredPermission}</code>
            </p>
            <div className="mt-6">
              <p className="text-sm text-gray-600">
                Your roles: {user.roles.join(', ')}
              </p>
            </div>
          </div>
        </div>
      );
    }
  }

  // User is authenticated and has required permission
  return <>{children}</>;
};

// Helper function to check if user has a specific permission
function checkUserPermission(user: { roles: string[] }, permission: Permission): boolean {
  // Define role permissions (should match backend RBAC system)
  const rolePermissions: Record<string, Permission[]> = {
    admin: [
      "user:create", "user:read", "user:update", "user:delete",
      "agent:create", "agent:read", "agent:update", "agent:delete",
      "system:admin", "system:manage", "system:read", "websocket:connect"
    ],
    manager: [
      "user:read", "user:update",
      "agent:create", "agent:read", "agent:update",
      "system:manage", "websocket:connect"
    ],
    user: [
      "agent:read", "agent:create",
      "websocket:connect"
    ],
    guest: [
      "agent:read"
    ]
  };

  // Check if any of the user's roles have the required permission
  return user.roles.some(role => {
    const permissions = rolePermissions[role] || [];
    return permissions.includes(permission);
  });
}

// Higher-order component for protecting components
export function withAuth<P extends object>(
  Component: React.ComponentType<P>,
  requiredPermission?: Permission
) {
  const WrappedComponent = (props: P) => (
    <ProtectedRoute requiredPermission={requiredPermission}>
      <Component {...props} />
    </ProtectedRoute>
  );
  
  WrappedComponent.displayName = `withAuth(${Component.displayName || Component.name})`;
  
  return WrappedComponent;
}

// Hook for checking permissions
export const usePermissions = () => {
  const { user } = useAuth();

  const hasPermission = (permission: Permission): boolean => {
    if (!user) return false;
    return checkUserPermission(user, permission);
  };

  const hasAnyPermission = (permissions: Permission[]): boolean => {
    return permissions.some(permission => hasPermission(permission));
  };

  const hasAllPermissions = (permissions: Permission[]): boolean => {
    return permissions.every(permission => hasPermission(permission));
  };

  return {
    hasPermission,
    hasAnyPermission,
    hasAllPermissions,
    userRoles: user?.roles || [],
  };
};