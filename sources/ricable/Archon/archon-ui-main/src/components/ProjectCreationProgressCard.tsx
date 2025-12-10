import React, { useState } from 'react';
import { Card } from './ui/Card';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  CheckCircle, 
  XCircle, 
  Loader2, 
  FileText, 
  ChevronDown, 
  ChevronUp, 
  RotateCcw,
  Clock,
  Bot,
  BrainCircuit,
  BookOpen,
  Database,
  AlertCircle
} from 'lucide-react';
import { Button } from './ui/Button';
import { ProjectCreationProgressData } from '../services/projectCreationProgressService';

interface ProjectCreationProgressCardProps {
  progressData: ProjectCreationProgressData;
  onComplete?: (data: ProjectCreationProgressData) => void;
  onError?: (error: string) => void;
  onRetry?: () => void;
  connectionStatus?: 'connected' | 'connecting' | 'disconnected' | 'error';
}

export const ProjectCreationProgressCard: React.FC<ProjectCreationProgressCardProps> = ({
  progressData,
  onComplete,
  onError,
  onRetry,
  connectionStatus = 'connected'
}) => {
  const [showLogs, setShowLogs] = useState(false);
  const [hasCompletedRef] = useState({ value: false });
  const [hasErroredRef] = useState({ value: false });

  // Handle completion/error events
  React.useEffect(() => {
    if (progressData.status === 'completed' && onComplete && !hasCompletedRef.value) {
      hasCompletedRef.value = true;
      onComplete(progressData);
    } else if (progressData.status === 'error' && onError && !hasErroredRef.value) {
      hasErroredRef.value = true;
      onError(progressData.error || 'Project creation failed');
    }
  }, [progressData.status, onComplete, onError, progressData, hasCompletedRef, hasErroredRef]);

  const getStatusIcon = () => {
    switch (progressData.status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'initializing_agents':
        return <Bot className="w-5 h-5 text-blue-500 animate-pulse" />;
      case 'generating_docs':
      case 'processing_requirements':
      case 'ai_generation':
        return <BrainCircuit className="w-5 h-5 text-purple-500 animate-pulse" />;
      case 'finalizing_docs':
        return <BookOpen className="w-5 h-5 text-indigo-500 animate-pulse" />;
      case 'saving_to_database':
        return <Database className="w-5 h-5 text-green-500 animate-pulse" />;
      default:
        return <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />;
    }
  };

  const getStatusColor = () => {
    switch (progressData.status) {
      case 'completed':
        return 'text-green-500';
      case 'error':
        return 'text-red-500';
      case 'initializing_agents':
        return 'text-blue-500';
      case 'generating_docs':
      case 'processing_requirements':
      case 'ai_generation':
        return 'text-purple-500';
      case 'finalizing_docs':
        return 'text-indigo-500';
      case 'saving_to_database':
        return 'text-green-500';
      default:
        return 'text-blue-500';
    }
  };

  const getStatusText = () => {
    switch (progressData.status) {
      case 'starting':
        return 'Starting project creation...';
      case 'initializing_agents':
        return 'Initializing AI agents...';
      case 'generating_docs':
        return 'Generating documentation...';
      case 'processing_requirements':
        return 'Processing requirements...';
      case 'ai_generation':
        return 'AI is creating project docs...';
      case 'finalizing_docs':
        return 'Finalizing documents...';
      case 'saving_to_database':
        return 'Saving to database...';
      case 'completed':
        return 'Project created successfully!';
      case 'error':
        return 'Project creation failed';
      default:
        return 'Processing...';
    }
  };

  const isActive = progressData.status !== 'completed' && progressData.status !== 'error';

  return (
    <Card className="p-6 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          {getStatusIcon()}
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white">
              Creating Project: {progressData.project?.title || 'New Project'}
            </h3>
            <p className={`text-sm ${getStatusColor()}`}>
              {getStatusText()}
            </p>
          </div>
        </div>
        
        {progressData.eta && isActive && (
          <div className="flex items-center gap-1 text-sm text-gray-500 dark:text-gray-400">
            <Clock className="w-4 h-4" />
            <span>{progressData.eta}</span>
          </div>
        )}
      </div>

      {/* Connection Status Indicator */}
      {connectionStatus !== 'connected' && (
        <div className="mb-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
          <div className="flex items-center gap-2 text-sm text-yellow-700 dark:text-yellow-400">
            {connectionStatus === 'connecting' && <Loader2 className="w-4 h-4 animate-spin" />}
            {connectionStatus === 'disconnected' && <AlertCircle className="w-4 h-4" />}
            {connectionStatus === 'error' && <XCircle className="w-4 h-4" />}
            <span>
              {connectionStatus === 'connecting' && 'Connecting to progress stream...'}
              {connectionStatus === 'disconnected' && 'Disconnected from progress stream'}
              {connectionStatus === 'error' && 'Connection error - retrying...'}
            </span>
          </div>
        </div>
      )}

      {/* Progress Bar */}
      <div className="mb-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm text-gray-600 dark:text-gray-400">
            Progress
          </span>
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            {progressData.percentage}%
          </span>
        </div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
          <motion.div
            className={`h-2 rounded-full transition-all duration-500 ${
              progressData.status === 'error' 
                ? 'bg-red-500' 
                : progressData.status === 'completed'
                ? 'bg-green-500'
                : 'bg-purple-500'
            }`}
            initial={{ width: 0 }}
            animate={{ width: `${progressData.percentage}%` }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
          />
        </div>
      </div>

      {/* Step Information */}
      {progressData.step && (
        <div className="mb-4 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg">
          <div className="text-sm">
            <span className="text-gray-600 dark:text-gray-400">Current Step: </span>
            <span className="font-medium text-gray-900 dark:text-white">
              {progressData.step}
            </span>
          </div>
        </div>
      )}

      {/* Error Information */}
      {progressData.status === 'error' && (
        <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <div className="text-sm text-red-700 dark:text-red-400">
            <strong>Error:</strong> {progressData.error || 'Project creation failed'}
            {progressData.progressId && (
              <div className="mt-1 text-xs opacity-75">
                Progress ID: {progressData.progressId}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Debug Information - Show when stuck on starting status */}
      {progressData.status === 'starting' && progressData.percentage === 0 && connectionStatus === 'connected' && (
        <div className="mb-4 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
          <div className="text-sm text-blue-700 dark:text-blue-400">
            <strong>Debug:</strong> Connected to progress stream but no updates received yet.
            <div className="mt-1 text-xs opacity-75">
              Progress ID: {progressData.progressId}
            </div>
            <div className="mt-1 text-xs opacity-75">
              Check browser console for Socket.IO connection details.
            </div>
          </div>
        </div>
      )}

      {/* Duration (when completed) */}
      {progressData.status === 'completed' && progressData.duration && (
        <div className="mb-4 p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg">
          <div className="text-sm text-green-700 dark:text-green-400">
            <strong>Completed in:</strong> {progressData.duration}
          </div>
        </div>
      )}

      {/* Console Logs */}
      {progressData.logs && progressData.logs.length > 0 && (
        <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
          <button
            onClick={() => setShowLogs(!showLogs)}
            className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-white transition-colors mb-2"
          >
            <FileText className="w-4 h-4" />
            <span>View Console Output</span>
            {showLogs ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
          
          <AnimatePresence>
            {showLogs && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="bg-gray-900 dark:bg-black rounded-md p-3 max-h-32 overflow-y-auto">
                  <div className="space-y-1 font-mono text-xs">
                    {progressData.logs.map((log, index) => (
                      <div key={index} className="text-green-400">
                        {log}
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* Action Buttons */}
      {progressData.status === 'error' && onRetry && (
        <div className="flex justify-end mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
          <Button 
            onClick={onRetry}
            variant="primary" 
            accentColor="purple"
            className="text-sm"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Retry
          </Button>
        </div>
      )}
    </Card>
  );
}; 