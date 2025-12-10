// Custom hook for dashboard data fetching with retry logic and error handling
import { useState, useEffect, useCallback, useRef } from 'react';

interface UseDashboardDataOptions {
  autoRefresh?: boolean;
  refreshInterval?: number; // in milliseconds
  retryAttempts?: number;
  retryDelay?: number; // in milliseconds
  onError?: (error: Error) => void;
  onSuccess?: (data: any) => void;
}

interface UseDashboardDataState<T> {
  data: T | null;
  isLoading: boolean;
  error: string | null;
  isRetrying: boolean;
  lastFetch: Date | null;
  retryCount: number;
}

export function useDashboardData<T = any>(
  endpoint: string,
  options: UseDashboardDataOptions = {}
) {
  const {
    autoRefresh = true,
    refreshInterval = 30000, // 30 seconds
    retryAttempts = 3,
    retryDelay = 1000, // 1 second
    onError,
    onSuccess
  } = options;

  const [state, setState] = useState<UseDashboardDataState<T>>({
    data: null,
    isLoading: true,
    error: null,
    isRetrying: false,
    lastFetch: null,
    retryCount: 0
  });

  const abortControllerRef = useRef<AbortController | null>(null);
  const retryTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const refreshTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const fetchData = useCallback(async (isRetry = false) => {
    // Cancel any pending requests
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    // Create new abort controller for this request
    abortControllerRef.current = new AbortController();

    setState(prev => ({
      ...prev,
      isLoading: !isRetry,
      isRetrying: isRetry,
      error: isRetry ? prev.error : null
    }));

    try {
      const response = await fetch(endpoint, {
        signal: abortControllerRef.current.signal,
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      const now = new Date();

      setState(prev => ({
        ...prev,
        data,
        isLoading: false,
        isRetrying: false,
        error: null,
        lastFetch: now,
        retryCount: 0
      }));

      if (onSuccess) {
        onSuccess(data);
      }

      // Schedule next refresh if auto-refresh is enabled
      if (autoRefresh && refreshInterval > 0) {
        refreshTimeoutRef.current = setTimeout(() => {
          fetchData();
        }, refreshInterval);
      }

    } catch (error) {
      // Ignore abort errors
      if (error instanceof Error && error.name === 'AbortError') {
        return;
      }

      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      
      setState(prev => ({
        ...prev,
        isLoading: false,
        isRetrying: false,
        error: errorMessage,
        retryCount: prev.retryCount + 1
      }));

      if (onError) {
        onError(error instanceof Error ? error : new Error(errorMessage));
      }

      // Retry if we haven't exceeded the retry limit
      if (state.retryCount < retryAttempts) {
        const delay = retryDelay * Math.pow(2, state.retryCount); // Exponential backoff
        retryTimeoutRef.current = setTimeout(() => {
          fetchData(true);
        }, delay);
      }
    }
  }, [endpoint, autoRefresh, refreshInterval, retryAttempts, retryDelay, onError, onSuccess, state.retryCount]);

  const retry = useCallback(() => {
    setState(prev => ({ ...prev, retryCount: 0 }));
    fetchData(true);
  }, [fetchData]);

  const refresh = useCallback(() => {
    // Clear any pending refresh
    if (refreshTimeoutRef.current) {
      clearTimeout(refreshTimeoutRef.current);
      refreshTimeoutRef.current = null;
    }
    fetchData();
  }, [fetchData]);

  const pause = useCallback(() => {
    if (refreshTimeoutRef.current) {
      clearTimeout(refreshTimeoutRef.current);
      refreshTimeoutRef.current = null;
    }
    if (retryTimeoutRef.current) {
      clearTimeout(retryTimeoutRef.current);
      retryTimeoutRef.current = null;
    }
  }, []);

  const resume = useCallback(() => {
    if (autoRefresh && !refreshTimeoutRef.current && !state.isLoading && !state.isRetrying) {
      refresh();
    }
  }, [autoRefresh, refresh, state.isLoading, state.isRetrying]);

  // Initial fetch
  useEffect(() => {
    fetchData();

    return () => {
      // Cleanup on unmount
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
      if (refreshTimeoutRef.current) {
        clearTimeout(refreshTimeoutRef.current);
      }
    };
  }, [endpoint]); // Only re-run when endpoint changes

  // Cleanup timeouts on unmount
  useEffect(() => {
    return () => {
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current);
      }
      if (refreshTimeoutRef.current) {
        clearTimeout(refreshTimeoutRef.current);
      }
    };
  }, []);

  return {
    ...state,
    retry,
    refresh,
    pause,
    resume,
    canRetry: state.retryCount < retryAttempts && state.error !== null,
  };
}

// Specialized hook for multiple endpoints
export function useDashboardDataMultiple<T = any>(
  endpoints: string[],
  options: UseDashboardDataOptions = {}
) {
  const [state, setState] = useState<{
    data: Record<string, T | null>;
    isLoading: boolean;
    errors: Record<string, string | null>;
    lastFetch: Date | null;
  }>({
    data: {},
    isLoading: true,
    errors: {},
    lastFetch: null
  });

  const fetchAllData = useCallback(async () => {
    setState(prev => ({ ...prev, isLoading: true }));

    try {
      const promises = endpoints.map(async (endpoint) => {
        try {
          const response = await fetch(endpoint);
          if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
          }
          return { endpoint, data: await response.json(), error: null };
        } catch (error) {
          return { 
            endpoint, 
            data: null, 
            error: error instanceof Error ? error.message : 'Unknown error' 
          };
        }
      });

      const results = await Promise.all(promises);
      const now = new Date();

      const newData: Record<string, T | null> = {};
      const newErrors: Record<string, string | null> = {};

      results.forEach(({ endpoint, data, error }) => {
        newData[endpoint] = data;
        newErrors[endpoint] = error;
      });

      setState({
        data: newData,
        isLoading: false,
        errors: newErrors,
        lastFetch: now
      });

    } catch (error) {
      setState(prev => ({
        ...prev,
        isLoading: false,
        errors: endpoints.reduce((acc, endpoint) => ({
          ...acc,
          [endpoint]: error instanceof Error ? error.message : 'Unknown error'
        }), {})
      }));
    }
  }, [endpoints]);

  useEffect(() => {
    fetchAllData();

    if (options.autoRefresh && options.refreshInterval) {
      const interval = setInterval(fetchAllData, options.refreshInterval);
      return () => clearInterval(interval);
    }
  }, [fetchAllData, options.autoRefresh, options.refreshInterval]);

  return {
    ...state,
    refresh: fetchAllData,
    hasErrors: Object.values(state.errors).some(error => error !== null),
    errorCount: Object.values(state.errors).filter(error => error !== null).length
  };
}