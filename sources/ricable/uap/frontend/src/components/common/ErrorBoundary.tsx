// Enhanced error boundary and error display components for dashboard
import React, { Component, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Wifi, WifiOff } from 'lucide-react';

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: any;
}

interface ErrorBoundaryProps {
  children: ReactNode;
  fallbackComponent?: ReactNode;
  onError?: (error: Error, errorInfo: any) => void;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null
    };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return {
      hasError: true,
      error,
      errorInfo: null
    };
  }

  componentDidCatch(error: Error, errorInfo: any) {
    this.setState({
      error,
      errorInfo
    });

    // Call the onError callback if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Log error for debugging
    console.error('Dashboard Error Boundary caught an error:', error, errorInfo);
  }

  handleRetry = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null
    });
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallbackComponent) {
        return this.props.fallbackComponent;
      }

      return (
        <div className="p-6 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center">
            <AlertTriangle className="h-5 w-5 text-red-500 mr-2" />
            <h3 className="text-lg font-medium text-red-800">Something went wrong</h3>
          </div>
          <p className="text-red-700 mt-2">
            {this.state.error?.message || 'An unexpected error occurred in the dashboard component.'}
          </p>
          <button
            onClick={this.handleRetry}
            className="mt-4 flex items-center px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            <RefreshCw className="h-4 w-4 mr-2" />
            Try Again
          </button>
          {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
            <details className="mt-4">
              <summary className="text-sm text-red-600 cursor-pointer">
                View Error Details (Development)
              </summary>
              <pre className="mt-2 text-xs text-red-600 overflow-auto max-h-48">
                {this.state.error?.stack}
                {'\n\n'}
                {this.state.errorInfo.componentStack}
              </pre>
            </details>
          )}
        </div>
      );
    }

    return this.props.children;
  }
}

// Loading state component
interface LoadingStateProps {
  message?: string;
  showSpinner?: boolean;
}

export const LoadingState: React.FC<LoadingStateProps> = ({ 
  message = 'Loading...', 
  showSpinner = true 
}) => (
  <div className="p-6">
    <div className="animate-pulse">
      {showSpinner && (
        <div className="flex items-center justify-center mb-4">
          <div className="w-8 h-8 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
        </div>
      )}
      <div className="text-center text-gray-600">{message}</div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-6">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-gray-200 rounded-lg h-32"></div>
        ))}
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
        {[...Array(2)].map((_, i) => (
          <div key={i} className="bg-gray-200 rounded-lg h-64"></div>
        ))}
      </div>
    </div>
  </div>
);

// Error display component for API errors
interface ErrorDisplayProps {
  error: string;
  onRetry?: () => void;
  isRetrying?: boolean;
  showDetails?: boolean;
}

export const ErrorDisplay: React.FC<ErrorDisplayProps> = ({ 
  error, 
  onRetry, 
  isRetrying = false,
  showDetails = false 
}) => (
  <div className="p-6">
    <div className="bg-red-50 border border-red-200 rounded-lg p-4">
      <div className="flex items-center">
        <AlertTriangle className="h-5 w-5 text-red-500 mr-2" />
        <span className="text-red-700 font-medium">Error loading dashboard data</span>
      </div>
      <p className="text-red-600 mt-2">{error}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          disabled={isRetrying}
          className={`mt-3 flex items-center px-4 py-2 rounded-lg transition-colors ${
            isRetrying 
              ? 'bg-gray-400 text-gray-700 cursor-not-allowed' 
              : 'bg-red-600 text-white hover:bg-red-700'
          }`}
        >
          <RefreshCw className={`h-4 w-4 mr-2 ${isRetrying ? 'animate-spin' : ''}`} />
          {isRetrying ? 'Retrying...' : 'Retry'}
        </button>
      )}
      {showDetails && (
        <details className="mt-3">
          <summary className="text-sm text-red-600 cursor-pointer">View Technical Details</summary>
          <pre className="mt-2 text-xs text-red-600 bg-red-100 p-2 rounded overflow-auto max-h-32">
            {error}
          </pre>
        </details>
      )}
    </div>
  </div>
);

// Connection status indicator
interface ConnectionStatusProps {
  isConnected: boolean;
  isConnecting?: boolean;
  lastUpdate?: Date;
  error?: string;
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  isConnected,
  isConnecting = false,
  lastUpdate,
  error
}) => (
  <div className="flex items-center space-x-2 text-sm">
    {isConnecting ? (
      <>
        <div className="w-4 h-4 border-2 border-blue-200 border-t-blue-600 rounded-full animate-spin"></div>
        <span className="text-blue-600">Connecting...</span>
      </>
    ) : isConnected ? (
      <>
        <Wifi className="h-4 w-4 text-green-500" />
        <span className="text-green-600 font-medium">Live</span>
        {lastUpdate && (
          <span className="text-gray-500">
            Updated {lastUpdate.toLocaleTimeString()}
          </span>
        )}
      </>
    ) : (
      <>
        <WifiOff className="h-4 w-4 text-gray-400" />
        <span className="text-gray-500">
          {error ? 'Connection error' : 'Offline'}
        </span>
        {error && (
          <span className="text-red-500 text-xs">({error})</span>
        )}
      </>
    )}
  </div>
);

// Empty state component
interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description: string;
  action?: {
    label: string;
    onClick: () => void;
  };
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  action
}) => (
  <div className="text-center py-12">
    {icon && (
      <div className="flex justify-center mb-4">
        {icon}
      </div>
    )}
    <h3 className="text-lg font-medium text-gray-900 mb-2">{title}</h3>
    <p className="text-gray-500 mb-4">{description}</p>
    {action && (
      <button
        onClick={action.onClick}
        className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
      >
        {action.label}
      </button>
    )}
  </div>
);