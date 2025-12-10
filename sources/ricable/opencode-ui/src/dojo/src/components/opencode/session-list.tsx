/**
 * Session List - Enhanced session browser with thumbnails and metadata
 * Replicates Claudia's session list UI with OpenCode multi-provider support
 */

"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FileText,
  ArrowLeft,
  Search,
  Filter,
  Clock,
  MessageSquare,
  Share2,
  Trash2,
  Play,
  MoreHorizontal,
  Calendar,
  User,
  Activity,
  QrCode,
  ExternalLink,
  Archive,
  Star,
  ChevronRight,
  Zap
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuSeparator } from '@/components/ui/dropdown-menu';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { cn } from '@/lib/utils';
import { useSessionStore } from '@/lib/session-store';
import { Session } from '@/lib/opencode-client';

interface SessionListProps {
  /**
   * Array of sessions to display
   */
  sessions: Session[];
  /**
   * Current project path being viewed
   */
  projectPath: string;
  /**
   * Callback to go back to project list
   */
  onBack: () => void;
  /**
   * Callback when a session is clicked
   */
  onSessionClick?: (session: Session) => void;
  /**
   * Optional className for styling
   */
  className?: string;
}

interface SessionThumbnailProps {
  session: Session;
  isSelected?: boolean;
  onClick: () => void;
  onShare: () => void;
  onDelete: () => void;
  onArchive?: () => void;
}

const SessionThumbnail: React.FC<SessionThumbnailProps> = ({
  session,
  isSelected,
  onClick,
  onShare,
  onDelete,
  onArchive
}) => {
  const { providers } = useSessionStore();
  
  const currentProvider = providers.find(p => p.id === session.provider);
  
  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffInHours = Math.abs(now.getTime() - date.getTime()) / (1000 * 60 * 60);
    
    if (diffInHours < 24) {
      return date.toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit'
      });
    } else if (diffInHours < 24 * 7) {
      return date.toLocaleDateString('en-US', {
        weekday: 'short',
        hour: 'numeric',
        minute: '2-digit'
      });
    } else {
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined
      });
    }
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
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.2 }}
      className={cn(
        "group relative",
        isSelected && "ring-2 ring-primary"
      )}
    >
      <Card 
        className={cn(
          "cursor-pointer transition-all hover:shadow-md hover:scale-[1.02] active:scale-[0.98]",
          "border-l-4",
          getProviderColor(session.provider).replace('bg-', 'border-l-'),
          isSelected && "bg-muted/50"
        )}
        onClick={onClick}
      >
        <CardHeader className="pb-2">
          <div className="flex items-start justify-between">
            <div className="flex-1 min-w-0">
              <CardTitle className="text-sm font-medium truncate mb-1">
                {session.name || `Session ${session.id.slice(0, 8)}`}
              </CardTitle>
              <div className="flex items-center space-x-2">
                <div className={cn(
                  "h-2 w-2 rounded-full",
                  getProviderColor(session.provider)
                )} />
                <span className="text-xs text-muted-foreground">
                  {currentProvider?.name || session.provider}
                </span>
                <Badge 
                  variant="outline" 
                  className={cn(
                    "text-xs h-4 px-1",
                    session.status === 'active' && "border-green-500 text-green-700",
                    session.status === 'completed' && "border-blue-500 text-blue-700",
                    session.status === 'error' && "border-red-500 text-red-700"
                  )}
                >
                  {session.status}
                </Badge>
              </div>
            </div>
            
            <DropdownMenu>
              <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                <Button 
                  variant="ghost" 
                  size="icon" 
                  className="h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <MoreHorizontal className="h-3 w-3" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={(e) => {
                  e.stopPropagation();
                  onClick();
                }}>
                  <Play className="h-4 w-4 mr-2" />
                  Continue Session
                </DropdownMenuItem>
                <DropdownMenuItem onClick={(e) => {
                  e.stopPropagation();
                  onShare();
                }}>
                  <Share2 className="h-4 w-4 mr-2" />
                  Share Session
                </DropdownMenuItem>
                <DropdownMenuItem onClick={(e) => {
                  e.stopPropagation();
                  onArchive?.();
                }}>
                  <Archive className="h-4 w-4 mr-2" />
                  Archive
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem 
                  className="text-destructive"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete();
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
          <div className="space-y-3">
            {/* Session Preview - First/Last Message */}
            <div className="min-h-[2.5rem]">
              <p className="text-xs text-muted-foreground line-clamp-2">
                {session.preview_text || "No messages yet..."}
              </p>
            </div>
            
            {/* Metadata */}
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-1">
                  <MessageSquare className="h-3 w-3" />
                  <span>{session.message_count}</span>
                </div>
                <div className="flex items-center space-x-1">
                  <Clock className="h-3 w-3" />
                  <span>{formatDate(session.updated_at)}</span>
                </div>
              </div>
              
              <div className="flex items-center space-x-1">
                {session.total_cost > 0 && (
                  <span className="text-xs font-mono">
                    ${session.total_cost.toFixed(4)}
                  </span>
                )}
                {session.shared && (
                  <Share2 className="h-3 w-3 text-primary" />
                )}
              </div>
            </div>
            
            {/* Model Info */}
            <div className="flex items-center justify-between">
              <Badge variant="secondary" className="text-xs">
                {session.model}
              </Badge>
              {session.tools_used && session.tools_used.length > 0 && (
                <div className="flex items-center space-x-1">
                  <Zap className="h-3 w-3 text-muted-foreground" />
                  <span className="text-xs text-muted-foreground">
                    {session.tools_used.length} tools
                  </span>
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export const SessionList: React.FC<SessionListProps> = ({
  sessions,
  projectPath,
  onBack,
  onSessionClick,
  className
}) => {
  const { actions } = useSessionStore();
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'updated' | 'created' | 'name' | 'cost'>('updated');
  const [filterStatus, setFilterStatus] = useState<'all' | 'active' | 'completed' | 'error'>('all');
  const [selectedSession, setSelectedSession] = useState<Session | null>(null);
  const [showShareDialog, setShowShareDialog] = useState(false);
  const [shareUrl, setShareUrl] = useState('');
  const [qrCodeUrl, setQrCodeUrl] = useState('');

  // Filter and sort sessions
  const filteredSessions = sessions
    .filter(session => {
      const matchesSearch = searchQuery === '' || 
        session.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        session.id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        session.provider.toLowerCase().includes(searchQuery.toLowerCase());
      
      const matchesStatus = filterStatus === 'all' || session.status === filterStatus;
      
      return matchesSearch && matchesStatus;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'updated':
          return b.updated_at - a.updated_at;
        case 'created':
          return b.created_at - a.created_at;
        case 'name':
          return (a.name || a.id).localeCompare(b.name || b.id);
        case 'cost':
          return b.total_cost - a.total_cost;
        default:
          return 0;
      }
    });

  const handleSessionClick = (session: Session) => {
    actions.setActiveSession(session.id);
    actions.setCurrentView('session');
    onSessionClick?.(session);
  };

  const handleShareSession = async (session: Session) => {
    try {
      const url = await actions.shareSession(session.id);
      setShareUrl(url);
      setSelectedSession(session);
      // Generate QR code URL (using a service like qr-server.com)
      const qrUrl = `https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=${encodeURIComponent(url)}`;
      setQrCodeUrl(qrUrl);
      setShowShareDialog(true);
    } catch (error) {
      console.error('Failed to share session:', error);
    }
  };

  const handleDeleteSession = async (session: Session) => {
    if (confirm(`Are you sure you want to delete "${session.name || session.id}"?`)) {
      try {
        await actions.deleteSession(session.id);
      } catch (error) {
        console.error('Failed to delete session:', error);
      }
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      // Show success toast
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
    }
  };

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Header */}
      <div className="p-6 border-b border-border">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.3 }}
          className="flex items-center space-x-4 mb-4"
        >
          <Button
            variant="ghost"
            size="icon"
            onClick={onBack}
            className="h-8 w-8"
          >
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div className="flex-1 min-w-0">
            <h2 className="text-xl font-semibold truncate">{projectPath}</h2>
            <p className="text-sm text-muted-foreground">
              {sessions.length} session{sessions.length !== 1 ? 's' : ''}
              {filteredSessions.length !== sessions.length && ` (${filteredSessions.length} shown)`}
            </p>
          </div>
        </motion.div>

        {/* Search and Filters */}
        <div className="flex items-center space-x-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search sessions..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>
          
          <Select value={sortBy} onValueChange={(value: any) => setSortBy(value)}>
            <SelectTrigger className="w-[140px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="updated">Last Updated</SelectItem>
              <SelectItem value="created">Date Created</SelectItem>
              <SelectItem value="name">Name</SelectItem>
              <SelectItem value="cost">Cost</SelectItem>
            </SelectContent>
          </Select>
          
          <Select value={filterStatus} onValueChange={(value: any) => setFilterStatus(value)}>
            <SelectTrigger className="w-[120px]">
              <SelectValue />
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

      {/* Sessions Grid */}
      <div className="flex-1 overflow-auto p-6">
        {filteredSessions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-center">
            <FileText className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-medium mb-2">No sessions found</h3>
            <p className="text-muted-foreground">
              {searchQuery || filterStatus !== 'all'
                ? 'Try adjusting your search or filters'
                : 'No sessions in this project yet'
              }
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            <AnimatePresence>
              {filteredSessions.map((session) => (
                <SessionThumbnail
                  key={session.id}
                  session={session}
                  onClick={() => handleSessionClick(session)}
                  onShare={() => handleShareSession(session)}
                  onDelete={() => handleDeleteSession(session)}
                />
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* Share Dialog */}
      <Dialog open={showShareDialog} onOpenChange={setShowShareDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Share Session</DialogTitle>
            <DialogDescription>
              Share &quot;<span className="font-medium">{selectedSession?.name || selectedSession?.id}</span>&quot; with others
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            {/* QR Code */}
            {qrCodeUrl && (
              <div className="flex justify-center">
                <div className="p-4 bg-white rounded-lg">
                  <img src={qrCodeUrl} alt="Session QR Code" className="w-32 h-32" />
                </div>
              </div>
            )}
            
            {/* Share URL */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Share URL</label>
              <div className="flex space-x-2">
                <Input
                  value={shareUrl}
                  readOnly
                  className="flex-1 font-mono text-xs"
                />
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => copyToClipboard(shareUrl)}
                >
                  <ExternalLink className="h-4 w-4" />
                </Button>
              </div>
            </div>
            
            {/* Cross-device continuity info */}
            <div className="text-sm text-muted-foreground">
              <div className="flex items-center space-x-2 mb-2">
                <QrCode className="h-4 w-4" />
                <span className="font-medium">Cross-device continuity</span>
              </div>
              <p className="text-xs">
                Scan the QR code or use the link to continue this session on any device. 
                Changes will sync in real-time.
              </p>
            </div>
          </div>
          
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowShareDialog(false)}>
              Close
            </Button>
            <Button onClick={() => copyToClipboard(shareUrl)}>
              Copy Link
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};