"use client";

import React, { useState } from "react";
import { Topbar } from "@/components/topbar/topbar";
import { AnimatePresence } from "framer-motion";

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

interface OpenCodeLayoutProps {
  children: React.ReactNode;
}

export function OpenCodeLayout({ children }: OpenCodeLayoutProps) {
  const [currentView, setCurrentView] = useState<OpenCodeView>("welcome");

  return (
    <div className="h-screen bg-background flex flex-col">
      {/* Topbar Navigation */}
      <Topbar 
        currentView={currentView}
        onViewChange={setCurrentView}
      />
      
      {/* Main Content Area */}
      <div className="flex-1 overflow-hidden">
        <AnimatePresence mode="wait">
          {children}
        </AnimatePresence>
      </div>
    </div>
  );
}