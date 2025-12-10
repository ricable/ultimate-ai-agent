// Authentication page with login/register toggle
import React, { useState } from 'react';
import { LoginForm } from './LoginForm';
import { RegisterForm } from './RegisterForm';

interface AuthPageProps {
  onSuccess?: () => void;
  defaultMode?: 'login' | 'register';
  className?: string;
}

export const AuthPage: React.FC<AuthPageProps> = ({
  onSuccess,
  defaultMode = 'login',
  className = ''
}) => {
  const [mode, setMode] = useState<'login' | 'register'>(defaultMode);

  const handleSuccess = () => {
    onSuccess?.();
  };

  const switchToLogin = () => setMode('login');
  const switchToRegister = () => setMode('register');

  return (
    <div className={`min-h-screen bg-gray-100 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 ${className}`}>
      <div className="max-w-md w-full space-y-8">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900">
            UAP Platform
          </h1>
          <p className="mt-2 text-sm text-gray-600">
            Unified Agentic Platform
          </p>
        </div>

        {/* Auth Form */}
        <div className="bg-white rounded-lg shadow-lg">
          {mode === 'login' ? (
            <LoginForm
              onSuccess={handleSuccess}
              onSwitchToRegister={switchToRegister}
            />
          ) : (
            <RegisterForm
              onSuccess={handleSuccess}
              onSwitchToLogin={switchToLogin}
            />
          )}
        </div>

        {/* Footer */}
        <div className="text-center text-sm text-gray-500">
          <p>
            Secure authentication powered by JWT
          </p>
        </div>
      </div>
    </div>
  );
};