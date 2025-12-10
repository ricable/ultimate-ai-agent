/**
 * Project List - Enhanced project and session management interface
 * Ported from Claudia with multi-provider support for OpenCode
 */

"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FolderOpen, 
  Plus, 
  Search, 
  Filter,
  Clock,
  MessageSquare,
  Settings,
  Share2,
  Trash2,
  Play,
  MoreHorizontal,
  Calendar,
  User,
  Activity,
  Server
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuSeparator } from '@/components/ui/dropdown-menu';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import { useSessionStore } from '@/lib/session-store';
import { SessionConfig } from '@/lib/opencode-client';
import { SessionCreation } from './session-creation';

export const ProjectList: React.FC = () => {
  const {
    sessions,
    providers,
    isLoadingSessions,
    actions
  } = useSessionStore();

  const [searchQuery, setSearchQuery] = useState('');
  const [filterProvider, setFilterProvider] = useState<string>('all');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [showNewSessionDialog, setShowNewSessionDialog] = useState(false);

  // Load sessions on component mount
  useEffect(() => {
    actions.loadSessions();
    actions.loadProviders();
  }, []);

  // Filter sessions based on search and filters
  const filteredSessions = sessions.filter(session => {
    const matchesSearch = searchQuery === '' || 
      session.project_path.toLowerCase().includes(searchQuery.toLowerCase()) ||
      session.name?.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesProvider = filterProvider === 'all' || session.provider === filterProvider;
    const matchesStatus = filterStatus === 'all' || session.status === filterStatus;
    
    return matchesSearch && matchesProvider && matchesStatus;
  });

  // Group sessions by project directory
  const groupedSessions = filteredSessions.reduce((groups, session) => {
    const key = session.project_path;
    if (!groups[key]) {
      groups[key] = [];
    }
    groups[key].push(session);
    return groups;
  }, {} as Record<string, typeof sessions>);

  const handleSessionCreated = (sessionId: string) => {
    actions.setActiveSession(sessionId);
    actions.setCurrentView('session');
  };

  const handleSessionClick = (sessionId: string) => {
    actions.setActiveSession(sessionId);
    actions.setCurrentView('session');
  };

  const handleDeleteSession = async (sessionId: string) => {
    if (confirm('Are you sure you want to delete this session?')) {
      try {
        await actions.deleteSession(sessionId);
      } catch (error) {
        console.error('Failed to delete session:', error);
      }
    }
  };

  const handleShareSession = async (sessionId: string) => {
    try {
      const shareUrl = await actions.shareSession(sessionId);
      await navigator.clipboard.writeText(shareUrl);
      // Show success message
    } catch (error) {
      console.error('Failed to share session:', error);
    }
  };

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getProviderColor = (providerId: string) => {
    const colors = {
      'anthropic': 'bg-orange-500',
      'openai': 'bg-green-500',
      'google': 'bg-blue-500',
      'groq': 'bg-purple-500',
    };
    return colors[providerId as keyof typeof colors] || 'bg-gray-500';
  };

  const getStatusColor = (status: string) => {
    const colors = {
      'active': 'bg-green-500',
      'completed': 'bg-blue-500',
      'error': 'bg-red-500'
    };
    return colors[status as keyof typeof colors] || 'bg-gray-500';
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-6 border-b border-border">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold">Projects & Sessions</h1>
            <p className="text-muted-foreground">
              Manage your AI coding sessions across multiple providers
            </p>
          </div>
          <Button onClick={() => setShowNewSessionDialog(true)}>
            <Plus className="h-4 w-4 mr-2" />
            New Session
          </Button>
        </div>

        {/* Search and Filters */}
        <div className="flex items-center space-x-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search sessions and projects..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
          
          <Select value={filterProvider} onValueChange={setFilterProvider}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="All Providers" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Providers</SelectItem>
              {providers.map(provider => (
                <SelectItem key={provider.id} value={provider.id}>
                  {provider.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select value={filterStatus} onValueChange={setFilterStatus}>
            <SelectTrigger className="w-[150px]">
              <SelectValue placeholder="All Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Status</SelectItem>
              <SelectItem value="active">Active</SelectItem>
              <SelectItem value="completed">Completed</SelectItem>
              <SelectItem value="error">Error</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {isLoadingSessions ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        ) : Object.keys(groupedSessions).length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-center">
            <FolderOpen className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">No sessions found</h3>
            <p className="text-muted-foreground mb-4">
              {searchQuery || filterProvider !== 'all' || filterStatus !== 'all'
                ? 'Try adjusting your search or filters'
                : 'Create your first AI coding session to get started'
              }
            </p>
            <Button onClick={() => setShowNewSessionDialog(true)}>
              <Plus className="h-4 w-4 mr-2" />
              Create First Session
            </Button>
          </div>
        ) : (
          <div className="space-y-6">
            {Object.entries(groupedSessions).map(([projectPath, projectSessions]) => (
              <motion.div
                key={projectPath}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="space-y-3"
              >
                {/* Project Header */}
                <div className="flex items-center space-x-3">
                  <FolderOpen className="h-5 w-5 text-muted-foreground" />
                  <h3 className="text-lg font-medium">{projectPath}</h3>
                  <Badge variant="outline">
                    {projectSessions.length} session{projectSessions.length !== 1 ? 's' : ''}
                  </Badge>
                </div>

                {/* Sessions Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {projectSessions.map((session) => (
                    <Card
                      key={session.id}
                      className="cursor-pointer transition-all hover:shadow-md hover:scale-[1.02]"
                      onClick={() => handleSessionClick(session.id)}
                    >
                      <CardHeader className="pb-3">
                        <div className="flex items-start justify-between">
                          <div className="flex-1 min-w-0">
                            <CardTitle className="text-base truncate">
                              {session.name || `Session ${session.id.slice(0, 8)}`}
                            </CardTitle>
                            <div className="flex items-center space-x-2 mt-1">
                              <div className={cn(
                                "h-2 w-2 rounded-full",
                                getProviderColor(session.provider)
                              )} />
                              <span className="text-xs text-muted-foreground">
                                {session.provider}
                              </span>
                              <Badge 
                                variant="outline" 
                                className={cn(
                                  "text-xs",
                                  getStatusColor(session.status),
                                  "text-white"
                                )}
                              >
                                {session.status}
                              </Badge>
                            </div>
                          </div>
                          
                          <DropdownMenu>
                            <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                              <Button variant="ghost" size="icon" className="h-8 w-8">
                                <MoreHorizontal className="h-4 w-4" />
                              </Button>
                            </DropdownMenuTrigger>
                            <DropdownMenuContent align="end">
                              <DropdownMenuItem onClick={(e) => {
                                e.stopPropagation();
                                handleSessionClick(session.id);
                              }}>
                                <Play className="h-4 w-4 mr-2" />
                                Continue Session
                              </DropdownMenuItem>
                              <DropdownMenuItem onClick={(e) => {
                                e.stopPropagation();
                                handleShareSession(session.id);
                              }}>
                                <Share2 className="h-4 w-4 mr-2" />
                                Share Session
                              </DropdownMenuItem>
                              <DropdownMenuSeparator />
                              <DropdownMenuItem 
                                className="text-destructive"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleDeleteSession(session.id);
                                }}
                              >
                                <Trash2 className="h-4 w-4 mr-2" />
                                Delete Session
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </div>
                      </CardHeader>
                      
                      <CardContent className="pt-0">
                        <div className="space-y-2">
                          <div className="flex items-center space-x-4 text-xs text-muted-foreground">
                            <div className="flex items-center space-x-1">
                              <MessageSquare className="h-3 w-3" />
                              <span>{session.message_count} messages</span>
                            </div>
                            <div className="flex items-center space-x-1">
                              <Clock className="h-3 w-3" />
                              <span>{formatDate(session.updated_at)}</span>
                            </div>
                          </div>
                          
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-muted-foreground">
                              Model: {session.model}
                            </span>
                            {session.token_usage && (
                              <span className="text-muted-foreground">
                                {session.token_usage.input_tokens + session.token_usage.output_tokens} tokens
                              </span>
                            )}
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>

      {/* Session Creation Dialog */}
      <SessionCreation
        open={showNewSessionDialog}
        onOpenChange={setShowNewSessionDialog}
        onSessionCreated={handleSessionCreated}
      />
    </div>
  );
};