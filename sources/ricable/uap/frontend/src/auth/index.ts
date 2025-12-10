// Authentication module exports
export { AuthProvider, useAuth, useAuthenticatedApi } from './AuthContext';
export { LoginForm } from './LoginForm';
export { RegisterForm } from './RegisterForm';
export { AuthPage } from './AuthPage';
export { ProtectedRoute, withAuth, usePermissions } from './ProtectedRoute';
export { UserProfile } from './UserProfile';
export { authApi, AuthApiError } from './api';

// Export types
export type {
  User,
  UserCreate,
  UserLogin,
  Token,
  AuthResponse,
  AuthContextType,
  Role,
  Permission,
  AuthError
} from './types';