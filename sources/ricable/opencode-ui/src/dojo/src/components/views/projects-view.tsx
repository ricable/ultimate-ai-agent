"use client";

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Plus,
  FolderOpen,
  ChevronRight,
  Clock,
  Loader2,
  ArrowLeft,
  Play,
  FileText
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useSessionStore } from "@/lib/session-store";
import { OpenCodeView } from "@/types/opencode";

interface Project {
  id: string;
  name: string;
  path: string;
  sessions: Session[];
  created_at: number;
  last_modified: number;
  description?: string;
}

interface Session {
  id: string;
  name: string;
  created_at: number;
  provider: string;
  model: string;
  status: "active" | "completed" | "error";
  message_count: number;
  cost: number;
}

interface ProjectsViewProps {
  onViewChange: (view: OpenCodeView) => void;
}

export function ProjectsView({ onViewChange }: ProjectsViewProps) {
  const { 
    sessions, 
    actions,
    activeSessionId
  } = useSessionStore();
  
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Mock data for demonstration
  useEffect(() => {
    const loadProjects = async () => {
      try {
        setLoading(true);
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        const mockProjects: Project[] = [
          {
            id: "1",
            name: "AI Chat Bot",
            path: "/Users/dev/projects/ai-chatbot",
            created_at: Date.now() - 86400000 * 7, // 7 days ago
            last_modified: Date.now() - 86400000 * 2, // 2 days ago
            description: "Building a sophisticated AI chatbot with multi-provider support",
            sessions: [
              {
                id: "s1",
                name: "Initial Setup",
                created_at: Date.now() - 86400000 * 7,
                provider: "anthropic",
                model: "claude-3.5-sonnet",
                status: "completed",
                message_count: 45,
                cost: 0.73
              },
              {
                id: "s2", 
                name: "API Integration",
                created_at: Date.now() - 86400000 * 3,
                provider: "openai",
                model: "gpt-4o",
                status: "completed",
                message_count: 28,
                cost: 0.45
              }
            ]
          },
          {
            id: "2",
            name: "React Dashboard",
            path: "/Users/dev/projects/dashboard",
            created_at: Date.now() - 86400000 * 14,
            last_modified: Date.now() - 86400000 * 1,
            description: "Modern dashboard with real-time analytics",
            sessions: [
              {
                id: "s3",
                name: "Component Architecture",
                created_at: Date.now() - 86400000 * 5,
                provider: "anthropic",
                model: "claude-3.5-haiku",
                status: "active",
                message_count: 12,
                cost: 0.23
              }
            ]
          }
        ];
        
        setProjects(mockProjects);
        setError(null);
      } catch (err) {
        setError("Failed to load projects");
        console.error("Error loading projects:", err);
      } finally {
        setLoading(false);
      }
    };

    loadProjects();
  }, []);

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getStatusColor = (status: Session["status"]) => {
    switch (status) {
      case "active":
        return "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200";
      case "completed":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200";
      case "error":
        return "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200";
      default:
        return "bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200";
    }
  };

  const handleNewSession = async () => {
    try {
      const sessionConfig = {
        project_path: "/tmp/quick-session",
        provider: "anthropic",
        model: "claude-3-5-sonnet-20241022",
        max_tokens: 8000,
        temperature: 0.7
      };
      
      const newSession = await actions.createSession(sessionConfig);
      actions.setActiveSession(newSession.id);
      onViewChange("session");
    } catch (error) {
      console.error("Failed to create session:", error);
    }
  };

  const handleSessionClick = (session: Session) => {
    actions.setActiveSession(session.id);
    onViewChange("session");
  };

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground mx-auto mb-2" />
          <p className="text-muted-foreground">Loading projects...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <p className="text-red-500 mb-4">{error}</p>
          <Button onClick={() => window.location.reload()}>
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Button 
              variant="ghost" 
              size="icon"
              onClick={() => onViewChange("welcome")}
            >
              <ArrowLeft className="h-4 w-4" />
            </Button>
            <div>
              <h1 className="text-2xl font-bold">
                {selectedProject ? selectedProject.name : "Projects"}
              </h1>
              <p className="text-sm text-muted-foreground">
                {selectedProject 
                  ? `Sessions for ${selectedProject.path}`
                  : "Manage your coding projects and sessions"
                }
              </p>
            </div>
          </div>
          <Button onClick={handleNewSession}>
            <Plus className="h-4 w-4 mr-2" />
            New Session
          </Button>
        </div>
      </div>

      <div className="p-6">
        <div className="max-w-4xl mx-auto">
          <AnimatePresence mode="wait">
            {selectedProject ? (
              // Session List View
              <motion.div
                key="sessions"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3 }}
              >
                {/* Back to Projects Button */}
                <div className="mb-6">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSelectedProject(null)}
                    className="mb-4"
                  >
                    <ArrowLeft className="h-4 w-4 mr-2" />
                    Back to Projects
                  </Button>
                  <div>
                    <h3 className="text-lg font-semibold mb-2">Sessions</h3>
                    {selectedProject.description && (
                      <p className="text-sm text-muted-foreground mb-4">
                        {selectedProject.description}
                      </p>
                    )}
                  </div>
                </div>

              {/* Sessions */}
              <div className="space-y-4">
                {selectedProject.sessions.map((session, index) => (
                  <motion.div
                    key={session.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ 
                      duration: 0.3,
                      delay: index * 0.1 
                    }}
                  >
                    <Card 
                      className="cursor-pointer hover:shadow-md transition-all hover:scale-[1.01]"
                      onClick={() => handleSessionClick(session)}
                    >
                      <CardContent className="p-4">
                        <div className="flex items-center justify-between">
                          <div className="flex-1">
                            <div className="flex items-center space-x-3 mb-2">
                              <Play className="h-4 w-4 text-muted-foreground" />
                              <h3 className="font-medium">{session.name}</h3>
                              <Badge className={getStatusColor(session.status)}>
                                {session.status}
                              </Badge>
                            </div>
                            <div className="flex items-center space-x-4 text-sm text-muted-foreground">
                              <span>{session.provider}/{session.model}</span>
                              <span>{session.message_count} messages</span>
                              <span>${session.cost.toFixed(2)}</span>
                              <div className="flex items-center space-x-1">
                                <Clock className="h-3 w-3" />
                                <span>{formatDate(session.created_at)}</span>
                              </div>
                            </div>
                          </div>
                          <ChevronRight className="h-4 w-4 text-muted-foreground" />
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          ) : (
            // Project List View
            <motion.div
              key="projects"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 20 }}
              transition={{ duration: 0.3 }}
            >
              {/* Header */}
              <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="mb-8"
              >
                <h1 className="text-3xl font-bold tracking-tight mb-2">Projects</h1>
                <p className="text-muted-foreground">
                  Manage your coding projects and OpenCode sessions
                </p>
              </motion.div>

              {/* New Session Button */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
                className="mb-6"
              >
                <Button onClick={handleNewSession} className="w-full">
                  <Plus className="h-4 w-4 mr-2" />
                  New OpenCode Session
                </Button>
              </motion.div>

              {/* Projects */}
              {projects.length > 0 ? (
                <div className="space-y-4">
                  {projects.map((project, index) => (
                    <motion.div
                      key={project.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ 
                        duration: 0.3,
                        delay: index * 0.1
                      }}
                    >
                      <Card 
                        className="cursor-pointer hover:shadow-md transition-all hover:scale-[1.01]"
                        onClick={() => setSelectedProject(project)}
                      >
                        <CardContent className="p-4">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center space-x-3 flex-1">
                              <FolderOpen className="h-5 w-5 text-muted-foreground" />
                              <div className="flex-1">
                                <h3 className="font-medium mb-1">{project.name}</h3>
                                <p className="text-sm text-muted-foreground mb-2">
                                  {project.path}
                                </p>
                                {project.description && (
                                  <p className="text-xs text-muted-foreground mb-2">
                                    {project.description}
                                  </p>
                                )}
                                <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                                  <span>
                                    {project.sessions.length} session{project.sessions.length !== 1 ? 's' : ''}
                                  </span>
                                  <div className="flex items-center space-x-1">
                                    <Clock className="h-3 w-3" />
                                    <span>{formatDate(project.last_modified)}</span>
                                  </div>
                                </div>
                              </div>
                            </div>
                            <ChevronRight className="h-4 w-4 text-muted-foreground" />
                          </div>
                        </CardContent>
                      </Card>
                    </motion.div>
                  ))}
                </div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                  className="text-center py-12"
                >
                  <FolderOpen className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="text-lg font-medium mb-2">No projects found</h3>
                  <p className="text-muted-foreground mb-4">
                    Create your first OpenCode session to get started
                  </p>
                  <Button onClick={handleNewSession}>
                    <Plus className="h-4 w-4 mr-2" />
                    New Session
                  </Button>
                </motion.div>
              )}
            </motion.div>
          )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}