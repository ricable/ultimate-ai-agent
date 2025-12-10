"use client";

import { SidebarProvider, SidebarInset } from "@/components/ui/sidebar";
import { AppSidebar } from "./app-sidebar";
import { AgentsProvider } from "@/providers/Agents";
import { MCPProvider } from "@/providers/MCP";
import { RagProvider } from "@/features/rag/providers/RAG";

export function SidebarLayout({ children }: { children: React.ReactNode }) {
  return (
    <SidebarProvider>
      <MCPProvider>
        <AgentsProvider>
          <RagProvider>
            <AppSidebar />
            <SidebarInset>{children}</SidebarInset>
          </RagProvider>
        </AgentsProvider>
      </MCPProvider>
    </SidebarProvider>
  );
}
