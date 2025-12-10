// CopilotKit provider wrapper for the application
import React from 'react';
import { CopilotKit } from '@copilotkit/react-core';

interface CopilotProviderProps {
  children: React.ReactNode;
}

// Backend runtime URL for CopilotKit
const COPILOT_RUNTIME_URL = '/api/copilot';

export const CopilotProvider: React.FC<CopilotProviderProps> = ({ children }) => {
  return (
    <CopilotKit runtimeUrl={COPILOT_RUNTIME_URL}>
      {children}
    </CopilotKit>
  );
};