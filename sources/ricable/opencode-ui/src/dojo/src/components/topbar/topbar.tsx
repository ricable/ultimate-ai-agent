"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { 
  Settings, 
  Bot, 
  FolderOpen, 
  BarChart3, 
  Plug, 
  Wrench,
  Layers3,
  Home
} from "lucide-react";
import { OpenCodeView } from "@/components/layout/opencode-layout";
import { cn } from "@/lib/utils";

interface TopbarProps {
  currentView: OpenCodeView;
  onViewChange: (view: OpenCodeView) => void;
}

export function Topbar({ currentView, onViewChange }: TopbarProps) {
  const navItems = [
    { id: "welcome" as const, icon: Home, label: "Home" },
    { id: "projects" as const, icon: FolderOpen, label: "Projects" },
    { id: "providers" as const, icon: Layers3, label: "Providers" },
    { id: "agents" as const, icon: Bot, label: "Agents" },
    { id: "tools" as const, icon: Wrench, label: "Tools" },
    { id: "mcp" as const, icon: Plug, label: "MCP" },
    { id: "usage-dashboard" as const, icon: BarChart3, label: "Usage" },
    { id: "settings" as const, icon: Settings, label: "Settings" },
  ];

  return (
    <div className="h-12 border-b border-border flex items-center justify-between px-4">
      {/* Left: Logo and Navigation */}
      <div className="flex items-center space-x-1">
        <div className="text-lg font-bold text-primary mr-4">
          OpenCode Desktop
        </div>
        
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <Button
              key={item.id}
              variant={currentView === item.id ? "default" : "ghost"}
              size="sm"
              onClick={() => onViewChange(item.id)}
              className={cn(
                "h-8 px-3 text-xs",
                currentView === item.id && "bg-primary text-primary-foreground"
              )}
            >
              <Icon className="h-3 w-3 mr-1" />
              {item.label}
            </Button>
          );
        })}
      </div>

      {/* Right: Theme Toggle */}
      <div className="flex items-center space-x-2">
        <ThemeToggle />
      </div>
    </div>
  );
}