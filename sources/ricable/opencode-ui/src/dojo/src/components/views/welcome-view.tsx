"use client";

import React, { useEffect } from "react";
import { motion } from "framer-motion";
import { 
  Bot, 
  FolderCode, 
  Layers3, 
  Wrench, 
  Plug, 
  BarChart3,
  Settings,
  Sparkles
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useSessionStore } from "@/lib/session-store";
import { OpenCodeView } from "@/types/opencode";

interface WelcomeViewProps {
  onViewChange: (view: OpenCodeView) => void;
}

export function WelcomeView({ onViewChange }: WelcomeViewProps) {
  const { 
    sessions, 
    providers, 
    authenticatedProviders, 
    serverStatus,
    actions 
  } = useSessionStore();

  useEffect(() => {
    // Load data on mount
    actions.loadSessions();
    actions.loadProviders();
  }, []);

  // Main action cards (like Claudia)
  const mainCards = [
    {
      id: "agents" as const,
      title: "CC Agents",
      description: "Create and manage custom AI agents",
      icon: Bot,
      color: "text-white",
      bgColor: "bg-gradient-to-br from-gray-800 to-gray-900 dark:from-gray-900 dark:to-black",
      iconBg: "bg-white/10",
    },
    {
      id: "projects" as const,
      title: "CC Projects", 
      description: "Manage your coding projects and sessions",
      icon: FolderCode,
      color: "text-white",
      bgColor: "bg-gradient-to-br from-gray-800 to-gray-900 dark:from-gray-900 dark:to-black",
      iconBg: "bg-white/10",
    },
  ];

  // Secondary navigation cards
  const navigationCards = [
    {
      id: "providers" as const,
      title: "AI Providers",
      description: `${authenticatedProviders.length}/${providers.length} providers configured`,
      icon: Layers3,
      color: "text-blue-500",
      bgColor: "bg-blue-50 dark:bg-blue-950/20",
      badge: authenticatedProviders.length > 0 ? authenticatedProviders.length : null,
    },
    {
      id: "tools" as const,
      title: "Tools",
      description: "File operations, system tools, and more",
      icon: Wrench,
      color: "text-orange-500",
      bgColor: "bg-orange-50 dark:bg-orange-950/20",
    },
    {
      id: "mcp" as const,
      title: "MCP Servers",
      description: "Model Context Protocol server management",
      icon: Plug,
      color: "text-cyan-500",
      bgColor: "bg-cyan-50 dark:bg-cyan-950/20",
    },
    {
      id: "usage-dashboard" as const,
      title: "Usage Analytics",
      description: `${sessions.length} sessions tracked`,
      icon: BarChart3,
      color: "text-rose-500",
      bgColor: "bg-rose-50 dark:bg-rose-950/20",
      badge: sessions.length > 0 ? sessions.length : null,
    },
  ];

  return (
    <div className="flex-1 overflow-y-auto bg-background">
      <div className="h-full flex flex-col">
        {/* Welcome Header - Matches Claudia's centered design */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="flex-1 flex flex-col items-center justify-center px-8"
        >
          <div className="text-center mb-16">
            <div className="flex items-center justify-center mb-6">
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-orange-400 to-red-500 flex items-center justify-center mr-4">
                <Sparkles className="h-6 w-6 text-white" />
              </div>
              <h1 className="text-4xl font-bold tracking-tight">
                Welcome to OpenCode
              </h1>
            </div>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-16">
              A powerful desktop GUI for OpenCode with support for 75+ AI providers. 
              Manage projects, configure agents, and leverage the full potential of multi-provider AI coding.
            </p>

            {/* Main Action Cards - Matches Claudia's two-card layout */}
            <div className="flex justify-center space-x-8 mb-16">
              {mainCards.map((card) => {
                const Icon = card.icon;
                return (
                  <motion.div
                    key={card.id}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Card 
                      className={`w-64 h-48 cursor-pointer transition-all duration-200 hover:shadow-xl border-0 ${card.bgColor}`}
                      onClick={() => onViewChange(card.id)}
                    >
                      <CardContent className="h-full flex flex-col items-center justify-center p-8 text-center">
                        <div className={`${card.iconBg} p-4 rounded-2xl mb-4`}>
                          <Icon className={`h-12 w-12 ${card.color}`} />
                        </div>
                        <h3 className={`text-xl font-semibold mb-2 ${card.color}`}>
                          {card.title}
                        </h3>
                        <p className={`text-sm opacity-80 ${card.color}`}>
                          {card.description}
                        </p>
                      </CardContent>
                    </Card>
                  </motion.div>
                );
              })}
            </div>

            {/* Status Information */}
            <div className="flex items-center justify-center space-x-6 text-sm text-muted-foreground">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  serverStatus === 'connected' ? 'bg-green-500' : 
                  serverStatus === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
                }`}></div>
                <span>Server: {serverStatus}</span>
              </div>
              <div className="flex items-center space-x-2">
                <span>{providers.length} providers available</span>
              </div>
              <div className="flex items-center space-x-2">
                <span>{sessions.length} active sessions</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Secondary Navigation - Bottom section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="px-8 pb-8"
        >
          <div className="max-w-4xl mx-auto">
            <h3 className="text-lg font-semibold mb-6 text-center">Quick Access</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {navigationCards.map((card, index) => {
                const Icon = card.icon;
                return (
                  <Card 
                    key={card.id}
                    className="cursor-pointer transition-all duration-200 hover:shadow-md hover:scale-105 border border-border/50 group"
                    onClick={() => onViewChange(card.id)}
                  >
                    <CardContent className="p-4 text-center">
                      <div className={`${card.bgColor} p-2 rounded-lg mb-2 group-hover:scale-110 transition-transform duration-200 mx-auto w-fit relative`}>
                        <Icon className={`h-4 w-4 ${card.color}`} />
                        {card.badge && (
                          <Badge 
                            variant="destructive" 
                            className="absolute -top-1 -right-1 h-4 w-4 p-0 text-xs flex items-center justify-center"
                          >
                            {card.badge}
                          </Badge>
                        )}
                      </div>
                      <h4 className="text-sm font-medium mb-1">{card.title}</h4>
                      <p className="text-xs text-muted-foreground leading-tight">
                        {card.description}
                      </p>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}