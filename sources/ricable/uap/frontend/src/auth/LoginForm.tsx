// Login form component
import React, { useState } from 'react';
import { useAuth } from './AuthContext';
import { UserLogin } from './types';

interface LoginFormProps {
  onSuccess?: () => void;
  onSwitchToRegister?: () => void;
  className?: string;
}

export const LoginForm: React.FC<LoginFormProps> = ({
  onSuccess,
  onSwitchToRegister,
  className = ''
}) => {
  const { login, isLoading, error } = useAuth();
  const [formData, setFormData] = useState<UserLogin>({
    username: '',
    password: ''
  });
  const [validationErrors, setValidationErrors] = useState<Partial<UserLogin>>({});

  const validateForm = (): boolean => {
    const errors: Partial<UserLogin> = {};

    if (!formData.username.trim()) {
      errors.username = 'Username is required';
    } else if (formData.username.length < 3) {
      errors.username = 'Username must be at least 3 characters';
    }

    if (!formData.password) {
      errors.password = 'Password is required';
    } else if (formData.password.length < 8) {
      errors.password = 'Password must be at least 8 characters';
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    try {
      await login(formData);
      onSuccess?.();
    } catch (error) {
      // Error is handled by the auth context
      console.error('Login error:', error);
    }
  };

  const handleInputChange = (field: keyof UserLogin) => (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    setFormData(prev => ({
      ...prev,
      [field]: e.target.value
    }));

    // Clear validation error when user starts typing
    if (validationErrors[field]) {
      setValidationErrors(prev => ({
        ...prev,
        [field]: undefined
      }));
    }
  };

  const getInputClassName = (field: keyof UserLogin) => {
    const baseClasses = "w-full px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500";
    const errorClasses = validationErrors[field] ? "border-red-500" : "border-gray-300";
    return `${baseClasses} ${errorClasses}`;
  };

  return (
    <div className={`max-w-md mx-auto p-6 bg-white rounded-lg shadow-md ${className}`}>
      <h2 className="text-2xl font-bold text-center text-gray-800 mb-6">
        Login to UAP
      </h2>

      {error && (
        <div className="mb-4 p-3 text-sm text-red-700 bg-red-100 border border-red-300 rounded">
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label htmlFor="username" className="block text-sm font-medium text-gray-700 mb-1">
            Username
          </label>
          <input
            id="username"
            type="text"
            value={formData.username}
            onChange={handleInputChange('username')}
            className={getInputClassName('username')}
            placeholder="Enter your username"
            disabled={isLoading}
            autoComplete="username"
          />
          {validationErrors.username && (
            <p className="mt-1 text-sm text-red-600">{validationErrors.username}</p>
          )}
        </div>

        <div>
          <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1">
            Password
          </label>
          <input
            id="password"
            type="password"
            value={formData.password}
            onChange={handleInputChange('password')}
            className={getInputClassName('password')}
            placeholder="Enter your password"
            disabled={isLoading}
            autoComplete="current-password"
          />
          {validationErrors.password && (
            <p className="mt-1 text-sm text-red-600">{validationErrors.password}</p>
          )}
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className="w-full py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? 'Logging in...' : 'Login'}
        </button>
      </form>

      {onSwitchToRegister && (
        <div className="mt-6 text-center">
          <p className="text-sm text-gray-600">
            Don't have an account?{' '}
            <button
              type="button"
              onClick={onSwitchToRegister}
              className="font-medium text-blue-600 hover:text-blue-500 focus:outline-none focus:underline"
              disabled={isLoading}
            >
              Sign up
            </button>
          </p>
        </div>
      )}

      {/* Demo credentials info */}
      <div className="mt-4 p-3 bg-gray-50 rounded border text-sm text-gray-600">
        <p className="font-medium mb-1">Demo Credentials:</p>
        <p>Username: admin</p>
        <p>Password: admin123!</p>
      </div>
    </div>
  );
};