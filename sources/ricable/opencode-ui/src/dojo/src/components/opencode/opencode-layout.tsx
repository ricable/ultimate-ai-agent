/**
 * OpenCode Layout - Main layout component for the OpenCode desktop GUI
 * Replaces the demo viewer with a full-featured AI coding assistant interface
 * Based on Claudia's four-panel design with topbar navigation
 */

"use client";

import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Bot,
  FolderCode,
  Layers3,
  Wrench,
  Plug,
  BarChart3,
  Settings,
  Monitor,
  Home,
  Sparkles
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useSessionStore } from '@/lib/session-store';

// Import views
import { WelcomeView } from '@/components/views/welcome-view';
import { ProjectsView } from '@/components/views/projects-view';
import { ProvidersView } from '@/components/views/providers-view';
import { AgentsView } from '@/components/views/agents-view';
import { ToolsView } from '@/components/views/tools-view';
import { MCPView } from '@/components/views/mcp-view';
import { UsageDashboardView } from '@/components/views/usage-dashboard-view';
import { SettingsView } from '@/components/views/settings-view';
import { SessionView } from '@/components/views/session-view';

interface OpenCodeLayoutProps {
  children?: React.ReactNode;
}

export type OpenCodeView = 
  | "welcome" 
  | "projects" 
  | "providers" 
  | "agents" 
  | "settings" 
  | "session" 
  | "usage-dashboard" 
  | "mcp"
  | "tools";

export const OpenCodeLayout: React.FC<OpenCodeLayoutProps> = ({ children }) => {
  const {
    serverStatus,
    activeSessionId,
    pendingApprovals,
    actions
  } = useSessionStore();

  const [currentView, setCurrentView] = useState<OpenCodeView>("welcome");
  const [isInitialized, setIsInitialized] = useState(false);

  // Initialize connection on mount
  useEffect(() => {
    const initializeApp = async () => {
      try {
        await actions.connect();
        setIsInitialized(true);
      } catch (error) {
        console.error('Failed to initialize OpenCode:', error);
        setIsInitialized(true); // Still show UI even if connection fails
      }
    };

    initializeApp();

    // Cleanup on unmount
    return () => {
      actions.disconnect();
    };
  }, []);

  const topbarItems = [
    {
      id: 'home',
      label: 'OpenCode 1.0',
      icon: Sparkles,
      view: 'welcome' as const,
      style: 'brand'
    },
    {
      id: 'usage-dashboard',
      label: 'Usage Dashboard',
      icon: BarChart3,
      view: 'usage-dashboard' as const,
      style: 'nav'
    },
    {
      id: 'projects',
      label: 'Projects',
      icon: FolderCode,
      view: 'projects' as const,
      style: 'nav'
    },
    {
      id: 'providers',
      label: 'Providers',
      icon: Layers3,
      view: 'providers' as const,
      style: 'nav'
    },
    {
      id: 'mcp',
      label: 'MCP',
      icon: Plug,
      view: 'mcp' as const,
      style: 'nav'
    },
    {
      id: 'settings',
      label: 'Settings',
      icon: Settings,
      view: 'settings' as const,
      style: 'nav'
    }
  ];

  const renderMainContent = () => {
    switch (currentView) {
      case 'welcome':
        return <WelcomeView onViewChange={setCurrentView} />;
      case 'projects':
        return <ProjectsView onViewChange={setCurrentView} />;
      case 'providers':
        return <ProvidersView onViewChange={setCurrentView} />;
      case 'agents':
        return <AgentsView onViewChange={setCurrentView} />;
      case 'tools':
        return <ToolsView onViewChange={setCurrentView} />;
      case 'mcp':
        return <MCPView onViewChange={setCurrentView} />;
      case 'usage-dashboard':
        return <UsageDashboardView onViewChange={setCurrentView} />;
      case 'settings':
        return <SettingsView onViewChange={setCurrentView} />;
      case 'session':
        return activeSessionId ? (
          <SessionView sessionId={activeSessionId} onViewChange={setCurrentView} />
        ) : (
          <WelcomeView onViewChange={setCurrentView} />
        );
      default:
        return <WelcomeView onViewChange={setCurrentView} />;
    }
  };

  if (!isInitialized) {
    return (
      <div className="flex items-center justify-center h-screen bg-background">
        <div className="flex flex-col items-center space-y-4">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          <p className="text-sm text-muted-foreground">Connecting to OpenCode...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen bg-background flex flex-col">
      {/* Topbar Navigation - Matches Claudia's design */}
      <div className="border-b border-border bg-card">
        <div className="flex items-center justify-between px-4 py-2">
          {/* Left side - Brand */}
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="text-sm font-medium">OpenCode 1.0.35</span>
            </div>
          </div>

          {/* Right side - Navigation */}
          <div className="flex items-center space-x-1">
            {topbarItems.slice(1).map((item) => {
              const Icon = item.icon;
              const isActive = currentView === item.view;

              return (
                <Button
                  key={item.id}
                  variant={isActive ? "secondary" : "ghost"}
                  size="sm"
                  onClick={() => setCurrentView(item.view)}
                  className="text-xs"
                >
                  <Icon className="h-3 w-3 mr-1" />
                  {item.label}
                </Button>
              );
            })}
          </div>
        </div>
      </div>
      
      {/* Main Content Area */}
      <div className="flex-1 overflow-hidden">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentView}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.2 }}
            className="h-full"
          >
            {renderMainContent()}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
};