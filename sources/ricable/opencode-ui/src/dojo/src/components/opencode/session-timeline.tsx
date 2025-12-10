/**
 * Session Timeline - Checkpoint management and timeline navigation
 * Provides visual timeline with checkpoint management and SQLite backend integration
 */

"use client";

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  GitBranch,
  Clock,
  MessageSquare,
  Bookmark,
  Save,
  RotateCcw,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Plus,
  Edit,
  Trash2,
  MoreHorizontal,
  CheckCircle,
  AlertCircle,
  Info,
  Zap,
  FileText,
  Code,
  Database,
  GitFork,
  ChevronDown,
  ChevronRight,
  Hash,
  FileCode,
  Diff,
  Settings,
  Loader2
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuSeparator } from '@/components/ui/dropdown-menu';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Textarea } from '@/components/ui/textarea';
import { cn } from '@/lib/utils';
import { useSessionStore, useActiveSession, useActiveSessionMessages } from '@/lib/session-store';
import { AdvancedCheckpointManager } from './advanced-checkpoint-manager';
import { 
  Checkpoint, 
  TimelineNode, 
  CheckpointDiff 
} from '@/types/opencode';

interface LocalSessionTimeline {
  rootNode: TimelineNode | null;
  currentCheckpointId: string | null;
  totalCheckpoints: number;
}

interface SessionTimelineProps {
  /**
   * Session ID to show timeline for
   */
  sessionId: string;
  /**
   * Whether the timeline is in sidebar mode
   */
  compact?: boolean;
  /**
   * Whether to use the advanced checkpoint manager
   */
  useAdvanced?: boolean;
  /**
   * Callback when a checkpoint is selected
   */
  onCheckpointSelect?: (checkpoint: Checkpoint) => void;
  /**
   * Optional className
   */
  className?: string;
}

interface TimelineNodeProps {
  node: TimelineNode;
  depth: number;
  timeline: LocalSessionTimeline;
  expandedNodes: Set<string>;
  selectedCheckpoint: Checkpoint | null;
  compact?: boolean;
  onSelect: (checkpoint: Checkpoint) => void;
  onToggleExpand: (nodeId: string) => void;
  onEdit: (checkpoint: Checkpoint) => void;
  onDelete: (checkpointId: string) => void;
  onRestore: (checkpoint: Checkpoint) => void;
  onFork: (checkpoint: Checkpoint) => void;
  onCompare: (checkpoint: Checkpoint) => void;
}

const TimelineNodeComponent: React.FC<TimelineNodeProps> = ({
  node,
  depth,
  timeline,
  expandedNodes,
  selectedCheckpoint,
  compact,
  onSelect,
  onToggleExpand,
  onEdit,
  onDelete,
  onRestore,
  onFork,
  onCompare
}) => {
  const { checkpoint } = node;
  const hasChildren = node.children.length > 0;
  const isExpanded = expandedNodes.has(checkpoint.id);
  const isCurrent = timeline?.currentCheckpointId === checkpoint.id;
  const isSelected = selectedCheckpoint?.id === checkpoint.id;

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffInMinutes = Math.abs(now.getTime() - date.getTime()) / (1000 * 60);
    
    if (diffInMinutes < 60) {
      return `${Math.floor(diffInMinutes)}m ago`;
    } else if (diffInMinutes < 24 * 60) {
      return `${Math.floor(diffInMinutes / 60)}h ago`;
    } else {
      return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric'
      });
    }
  };

  return (
    <div className="relative">
      {/* Connection line for child nodes */}
      {depth > 0 && (
        <div 
          className="absolute left-0 top-0 w-6 h-6 border-l-2 border-b-2 border-muted-foreground/30"
          style={{ 
            left: `${(depth - 1) * 24}px`,
            borderBottomLeftRadius: '8px'
          }}
        />
      )}
      
      {/* Node content */}
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.2, delay: depth * 0.05 }}
        className={cn(
          "flex items-start gap-2 py-2",
          depth > 0 && "ml-6"
        )}
        style={{ paddingLeft: `${depth * 24}px` }}
      >
        {/* Expand/collapse button */}
        {hasChildren && (
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 -ml-1 flex-shrink-0"
            onClick={() => onToggleExpand(checkpoint.id)}
          >
            {isExpanded ? (
              <ChevronDown className="h-3 w-3" />
            ) : (
              <ChevronRight className="h-3 w-3" />
            )}
          </Button>
        )}
        
        {/* Checkpoint card */}
        <Card 
          className={cn(
            "flex-1 cursor-pointer transition-all hover:shadow-md",
            isCurrent && "border-primary ring-2 ring-primary/20",
            isSelected && "border-blue-500 bg-blue-500/5",
            !hasChildren && "ml-5"
          )}
          onClick={() => onSelect(checkpoint)}
        >
          <CardContent className="p-3">
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  {isCurrent && (
                    <Badge variant="default" className="text-xs">Current</Badge>
                  )}
                  <span className="text-xs font-mono text-muted-foreground">
                    {checkpoint.id.slice(0, 8)}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {formatTime(checkpoint.timestamp || checkpoint.created_at)}
                  </span>
                </div>
                
                <p className="text-sm font-medium mb-1">
                  {checkpoint.name || checkpoint.description || 'Checkpoint'}
                </p>
                
                {checkpoint.description && checkpoint.name !== checkpoint.description && (
                  <p className="text-xs text-muted-foreground line-clamp-2 mb-2">
                    {checkpoint.description}
                  </p>
                )}
                
                {checkpoint.metadata.userPrompt && !compact && (
                  <p className="text-xs text-muted-foreground line-clamp-2 mb-2">
                    {checkpoint.metadata.userPrompt}
                  </p>
                )}
                
                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                  <span className="flex items-center gap-1">
                    <Hash className="h-3 w-3" />
                    {(checkpoint.metadata.totalTokens || 0).toLocaleString()} tokens
                  </span>
                  <span className="flex items-center gap-1">
                    <FileCode className="h-3 w-3" />
                    {checkpoint.metadata.fileChanges || 0} files
                  </span>
                  {checkpoint.metadata.provider && (
                    <span className="flex items-center gap-1">
                      <Database className="h-3 w-3" />
                      {checkpoint.metadata.provider}
                    </span>
                  )}
                </div>
              </div>
              
              {/* Actions */}
              <div className="flex items-center gap-1">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7"
                        onClick={(e) => {
                          e.stopPropagation();
                          onRestore(checkpoint);
                        }}
                      >
                        <RotateCcw className="h-3 w-3" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Restore to this checkpoint</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7"
                        onClick={(e) => {
                          e.stopPropagation();
                          onFork(checkpoint);
                        }}
                      >
                        <GitFork className="h-3 w-3" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Fork from this checkpoint</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7"
                        onClick={(e) => {
                          e.stopPropagation();
                          onCompare(checkpoint);
                        }}
                      >
                        <Diff className="h-3 w-3" />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Compare with another checkpoint</TooltipContent>
                  </Tooltip>
                </TooltipProvider>
                
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="ghost" size="icon" className="h-7 w-7">
                      <MoreHorizontal className="h-3 w-3" />
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent align="end">
                    <DropdownMenuItem onClick={(e) => { e.stopPropagation(); onEdit(checkpoint); }}>
                      <Edit className="h-4 w-4 mr-2" />
                      Edit Checkpoint
                    </DropdownMenuItem>
                    <DropdownMenuSeparator />
                    <DropdownMenuItem 
                      className="text-destructive"
                      onClick={(e) => { e.stopPropagation(); onDelete(checkpoint.id); }}
                    >
                      <Trash2 className="h-4 w-4 mr-2" />
                      Delete
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </div>
            
            {/* Type badge */}
            <div className="mt-2">
              <Badge 
                variant="outline" 
                className={cn(
                  "text-xs",
                  checkpoint.type === 'manual' && "border-blue-500 text-blue-700",
                  checkpoint.type === 'auto' && "border-gray-500 text-gray-700",
                  checkpoint.type === 'milestone' && "border-green-500 text-green-700"
                )}
              >
                {checkpoint.type}
              </Badge>
            </div>
          </CardContent>
        </Card>
      </motion.div>
      
      {/* Render children */}
      {isExpanded && hasChildren && (
        <div className="relative">
          {/* Vertical line for multiple children */}
          {node.children.length > 1 && (
            <div 
              className="absolute top-0 bottom-0 w-0.5 bg-muted-foreground/30"
              style={{ left: `${(depth + 1) * 24 - 1}px` }}
            />
          )}
          
          {node.children.map((child) => (
            <TimelineNodeComponent
              key={child.checkpoint.id}
              node={child}
              depth={depth + 1}
              timeline={timeline}
              expandedNodes={expandedNodes}
              selectedCheckpoint={selectedCheckpoint}
              compact={compact}
              onSelect={onSelect}
              onToggleExpand={onToggleExpand}
              onEdit={onEdit}
              onDelete={onDelete}
              onRestore={onRestore}
              onFork={onFork}
              onCompare={onCompare}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export const SessionTimeline: React.FC<SessionTimelineProps> = ({
  sessionId,
  compact = false,
  useAdvanced = false,
  onCheckpointSelect,
  className
}) => {
  const session = useActiveSession();
  const messages = useActiveSessionMessages();
  const { actions } = useSessionStore();
  
  const [timeline, setTimeline] = useState<LocalSessionTimeline | null>(null);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<Checkpoint | null>(null);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showEditDialog, setShowEditDialog] = useState(false);
  const [showDiffDialog, setShowDiffDialog] = useState(false);
  const [editingCheckpoint, setEditingCheckpoint] = useState<Checkpoint | null>(null);
  const [checkpointName, setCheckpointName] = useState('');
  const [checkpointDescription, setCheckpointDescription] = useState('');
  const [isCreating, setIsCreating] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [diff, setDiff] = useState<CheckpointDiff | null>(null);
  const [compareCheckpoint, setCompareCheckpoint] = useState<Checkpoint | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  // Helper function to create timeline tree structure
  const createTimelineTree = (checkpoints: Checkpoint[]): LocalSessionTimeline => {
    // Create a tree structure from flat checkpoint array
    // For now, create a simple tree with branching example
    const rootCheckpoint = checkpoints[0];
    const branchCheckpoint = checkpoints[1];
    const branchChild = checkpoints[2];
    const currentCheckpoint = checkpoints[3];

    if (!rootCheckpoint) {
      return {
        rootNode: null,
        currentCheckpointId: null,
        totalCheckpoints: 0
      };
    }

    // Create branch example: root -> [branch, current]
    const rootNode: TimelineNode = {
      checkpoint: rootCheckpoint,
      children: [],
      depth: 0
    };

    if (branchCheckpoint) {
      const branchNode: TimelineNode = {
        checkpoint: branchCheckpoint,
        children: branchChild ? [{
          checkpoint: branchChild,
          children: [],
          depth: 2,
          parent: undefined
        }] : [],
        depth: 1,
        parent: rootNode
      };
      
      rootNode.children.push(branchNode);
    }

    if (currentCheckpoint) {
      rootNode.children.push({
        checkpoint: currentCheckpoint,
        children: [],
        depth: 1,
        parent: rootNode
      });
    }

    return {
      rootNode,
      currentCheckpointId: currentCheckpoint?.id || null,
      totalCheckpoints: checkpoints.length
    };
  };

  const findPathToCheckpoint = useCallback((node: TimelineNode, checkpointId: string, path: string[] = []): string[] => {
    if (node.checkpoint.id === checkpointId) {
      return path;
    }
    
    for (const child of node.children) {
      const childPath = findPathToCheckpoint(child, checkpointId, [...path, node.checkpoint.id]);
      if (childPath.length > path.length) {
        return childPath;
      }
    }
    
    return path;
  }, []);

  // Load timeline on mount and whenever refreshVersion changes
  useEffect(() => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Mock checkpoints data - in real implementation this would come from SQLite
      const mockCheckpoints: Checkpoint[] = [
        {
          id: 'checkpoint-1',
          session_id: sessionId,
          name: 'Initial Setup',
          description: 'Started working on authentication refactor',
          timestamp: Date.now() - 3600000,
          created_at: Date.now() - 3600000,
          message_index: 0,
          type: 'auto',
          metadata: {
            totalTokens: 150,
            fileChanges: 0,
            cost: 0.001,
            provider: 'anthropic'
          }
        },
        {
          id: 'checkpoint-2',
          session_id: sessionId,
          name: 'Database Schema Updated',
          description: 'Modified user table to support new auth flow',
          timestamp: Date.now() - 2400000,
          created_at: Date.now() - 2400000,
          message_index: 5,
          type: 'manual',
          metadata: {
            totalTokens: 450,
            fileChanges: 2,
            cost: 0.012,
            toolsUsed: ['file_write', 'sql_executor'],
            filesModified: ['schema.sql', 'migrations/001_update_users.sql'],
            provider: 'openai'
          }
        },
        {
          id: 'checkpoint-3',
          session_id: sessionId,
          name: 'API Endpoints Implemented',
          description: 'Created login, register, and refresh token endpoints',
          timestamp: Date.now() - 1800000,
          created_at: Date.now() - 1800000,
          message_index: 8,
          type: 'milestone',
          metadata: {
            totalTokens: 680,
            fileChanges: 3,
            cost: 0.018,
            toolsUsed: ['file_write', 'bash'],
            filesModified: ['routes/auth.js', 'controllers/auth.js', 'middleware/jwt.js'],
            provider: 'openai'
          }
        },
        {
          id: 'checkpoint-4',
          session_id: sessionId,
          name: 'Current State',
          description: 'Working on frontend integration',
          timestamp: Date.now() - 300000,
          created_at: Date.now() - 300000,
          message_index: messages.length - 1,
          type: 'auto',
          metadata: {
            totalTokens: 1250,
            fileChanges: 2,
            cost: 0.045,
            toolsUsed: ['file_read', 'file_write'],
            filesModified: ['src/auth.js', 'src/api.js'],
            provider: 'anthropic'
          }
        }
      ];
      
      const timelineData = createTimelineTree(mockCheckpoints);
      setTimeline(timelineData);
      
      // Auto-expand nodes with current checkpoint
      if (timelineData.currentCheckpointId && timelineData.rootNode) {
        const pathToNode = findPathToCheckpoint(timelineData.rootNode, timelineData.currentCheckpointId);
        setExpandedNodes(new Set(pathToNode));
      }
    } catch (err) {
      console.error('Failed to load timeline:', err);
      setError('Failed to load timeline');
    } finally {
      setIsLoading(false);
    }
  }, [sessionId, messages.length, findPathToCheckpoint]);

  const toggleNodeExpansion = (nodeId: string) => {
    const newExpanded = new Set(expandedNodes);
    if (newExpanded.has(nodeId)) {
      newExpanded.delete(nodeId);
    } else {
      newExpanded.add(nodeId);
    }
    setExpandedNodes(newExpanded);
  };

  const handleCreateCheckpoint = async () => {
    if (!checkpointName.trim()) return;

    setIsCreating(true);
    try {
      const newCheckpoint: Checkpoint = {
        id: `checkpoint-${Date.now()}`,
        session_id: sessionId,
        name: checkpointName.trim(),
        description: checkpointDescription.trim() || undefined,
        timestamp: Date.now(),
        created_at: Date.now(),
        message_index: messages.length - 1,
        type: 'manual',
        metadata: {
          totalTokens: 0,
          fileChanges: 0,
          cost: 0,
          provider: 'current' // Get from active session
        }
      };

      // In real implementation, save to SQLite backend and reload timeline
      // For now, just close the dialog
      setShowCreateDialog(false);
      setCheckpointName('');
      setCheckpointDescription('');
    } catch (error) {
      console.error('Failed to create checkpoint:', error);
      setError('Failed to create checkpoint');
    } finally {
      setIsCreating(false);
    }
  };

  const handleEditCheckpoint = async () => {
    if (!editingCheckpoint || !checkpointName.trim()) return;

    setIsCreating(true);
    try {
      // In real implementation, update in SQLite backend and reload timeline
      setShowEditDialog(false);
      setEditingCheckpoint(null);
      setCheckpointName('');
      setCheckpointDescription('');
    } catch (error) {
      console.error('Failed to edit checkpoint:', error);
      setError('Failed to edit checkpoint');
    } finally {
      setIsCreating(false);
    }
  };

  const handleDeleteCheckpoint = async (checkpointId: string) => {
    if (confirm('Are you sure you want to delete this checkpoint?')) {
      try {
        // In real implementation, delete from SQLite backend and reload timeline
        console.log('Deleting checkpoint:', checkpointId);
      } catch (error) {
        console.error('Failed to delete checkpoint:', error);
        setError('Failed to delete checkpoint');
      }
    }
  };

  const handleRestoreCheckpoint = async (checkpoint: Checkpoint) => {
    if (confirm(`Restore to checkpoint "${checkpoint.description || checkpoint.name}"? Current state will be saved as a new checkpoint.`)) {
      try {
        setIsLoading(true);
        setError(null);
        
        // In real implementation:
        // 1. Create checkpoint of current state
        // 2. Restore to selected checkpoint
        // 3. Reload timeline
        console.log('Restoring to checkpoint:', checkpoint);
        onCheckpointSelect?.(checkpoint);
      } catch (error) {
        console.error('Failed to restore checkpoint:', error);
        setError('Failed to restore checkpoint');
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleFork = async (checkpoint: Checkpoint) => {
    try {
      // In real implementation, create a new branch from this checkpoint
      console.log('Forking from checkpoint:', checkpoint);
    } catch (error) {
      console.error('Failed to fork checkpoint:', error);
      setError('Failed to fork checkpoint');
    }
  };

  const handleCompare = async (checkpoint: Checkpoint) => {
    if (!selectedCheckpoint) {
      setSelectedCheckpoint(checkpoint);
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      
      // Mock diff data - in real implementation this would come from API
      const mockDiff: CheckpointDiff = {
        from_checkpoint: selectedCheckpoint,
        to_checkpoint: checkpoint,
        modified_files: [
          { 
            path: 'src/auth.js', 
            old_content: '// old content',
            new_content: '// new content',
            additions: 15, 
            deletions: 3,
            changes: []
          },
          { 
            path: 'src/api.js', 
            old_content: '// old content',
            new_content: '// new content',
            additions: 8, 
            deletions: 2,
            changes: []
          }
        ],
        added_files: ['src/utils.js'],
        deleted_files: [],
        token_delta: checkpoint.metadata.totalTokens - selectedCheckpoint.metadata.totalTokens,
        cost_delta: (checkpoint.metadata.cost || 0) - (selectedCheckpoint.metadata.cost || 0),
        total_changes: 25
      };
      
      setDiff(mockDiff);
      setCompareCheckpoint(checkpoint);
      setShowDiffDialog(true);
    } catch (error) {
      console.error('Failed to get diff:', error);
      setError('Failed to compare checkpoints');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectCheckpoint = (checkpoint: Checkpoint) => {
    setSelectedCheckpoint(checkpoint);
    onCheckpointSelect?.(checkpoint);
  };

  const openEditDialog = (checkpoint: Checkpoint) => {
    setEditingCheckpoint(checkpoint);
    setCheckpointName(checkpoint.name);
    setCheckpointDescription(checkpoint.description || '');
    setShowEditDialog(true);
  };

  const openCreateDialog = () => {
    setCheckpointName('');
    setCheckpointDescription('');
    setShowCreateDialog(true);
  };

  if (!session) {
    return (
      <div className="flex items-center justify-center h-32">
        <p className="text-sm text-muted-foreground">No session selected</p>
      </div>
    );
  }

  // Use advanced checkpoint manager if requested
  if (useAdvanced) {
    return (
      <AdvancedCheckpointManager
        sessionId={sessionId}
        compact={compact}
        onCheckpointSelect={onCheckpointSelect}
        className={className}
      />
    );
  }

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <GitBranch className="h-5 w-5" />
            <h3 className="font-medium">
              {compact ? 'Timeline' : 'Session Timeline'}
            </h3>
            {timeline && (
              <Badge variant="outline" className="text-xs">
                {timeline.totalCheckpoints} checkpoints
              </Badge>
            )}
          </div>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="outline"
                  size="icon"
                  onClick={openCreateDialog}
                  className="h-8 w-8"
                  disabled={isLoading}
                >
                  <Plus className="h-4 w-4" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>Create Checkpoint</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        
        {!compact && (
          <p className="text-xs text-muted-foreground">
            Track important moments in your session • 
            {selectedCheckpoint ? 'Click compare to see diff' : 'Select checkpoint to compare'}
          </p>
        )}
      </div>

      {/* Error display */}
      {error && (
        <div className="p-4 border-b border-border">
          <div className="flex items-center gap-2 text-xs text-destructive">
            <AlertCircle className="h-3 w-3" />
            {error}
          </div>
        </div>
      )}
      
      {/* Timeline */}
      <div className="flex-1 overflow-hidden">
        <ScrollArea ref={scrollAreaRef} className="h-full">
          <div className="p-4">
            {timeline?.rootNode ? (
              <div className="relative overflow-x-auto">
                <TimelineNodeComponent
                  node={timeline.rootNode}
                  depth={0}
                  timeline={timeline}
                  expandedNodes={expandedNodes}
                  selectedCheckpoint={selectedCheckpoint}
                  compact={compact}
                  onSelect={handleSelectCheckpoint}
                  onToggleExpand={toggleNodeExpansion}
                  onEdit={openEditDialog}
                  onDelete={handleDeleteCheckpoint}
                  onRestore={handleRestoreCheckpoint}
                  onFork={handleFork}
                  onCompare={handleCompare}
                />
              </div>
            ) : (
              <div className="text-center py-8">
                {isLoading ? (
                  <div className="flex flex-col items-center">
                    <Loader2 className="h-8 w-8 text-muted-foreground mx-auto mb-3 animate-spin" />
                    <p className="text-sm text-muted-foreground">Loading timeline...</p>
                  </div>
                ) : (
                  <div>
                    <Clock className="h-8 w-8 text-muted-foreground mx-auto mb-3" />
                    <p className="text-sm text-muted-foreground mb-2">
                      No checkpoints yet
                    </p>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={openCreateDialog}
                    >
                      <Plus className="h-4 w-4 mr-2" />
                      Create First Checkpoint
                    </Button>
                  </div>
                )}
              </div>
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Auto-checkpoint settings */}
      {!compact && (
        <div className="p-4 border-t border-border">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Auto-checkpoints every 10 messages</span>
            <Button variant="ghost" size="sm" className="h-6 text-xs">
              <Settings className="h-3 w-3 mr-1" />
              Configure
            </Button>
          </div>
        </div>
      )}

      {/* Create Checkpoint Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Create Checkpoint</DialogTitle>
            <DialogDescription>
              Save the current state of your session for easy navigation
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            <div>
              <Label htmlFor="checkpoint-name">Checkpoint Name *</Label>
              <Input
                id="checkpoint-name"
                placeholder="e.g., Implemented user authentication"
                value={checkpointName}
                onChange={(e) => setCheckpointName(e.target.value)}
              />
            </div>
            
            <div>
              <Label htmlFor="checkpoint-description">Description (Optional)</Label>
              <Textarea
                id="checkpoint-description"
                placeholder="Describe what was accomplished at this point..."
                value={checkpointDescription}
                onChange={(e) => setCheckpointDescription(e.target.value)}
                rows={3}
              />
            </div>
          </div>
          
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowCreateDialog(false)}
              disabled={isCreating}
            >
              Cancel
            </Button>
            <Button
              onClick={handleCreateCheckpoint}
              disabled={!checkpointName.trim() || isCreating}
            >
              {isCreating ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Creating...
                </>
              ) : (
                <>
                  <Save className="h-4 w-4 mr-2" />
                  Create Checkpoint
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Edit Checkpoint Dialog */}
      <Dialog open={showEditDialog} onOpenChange={setShowEditDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Edit Checkpoint</DialogTitle>
            <DialogDescription>
              Update checkpoint details
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            <div>
              <Label htmlFor="edit-checkpoint-name">Checkpoint Name *</Label>
              <Input
                id="edit-checkpoint-name"
                value={checkpointName}
                onChange={(e) => setCheckpointName(e.target.value)}
              />
            </div>
            
            <div>
              <Label htmlFor="edit-checkpoint-description">Description (Optional)</Label>
              <Textarea
                id="edit-checkpoint-description"
                value={checkpointDescription}
                onChange={(e) => setCheckpointDescription(e.target.value)}
                rows={3}
              />
            </div>
          </div>
          
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowEditDialog(false)}
              disabled={isCreating}
            >
              Cancel
            </Button>
            <Button
              onClick={handleEditCheckpoint}
              disabled={!checkpointName.trim() || isCreating}
            >
              {isCreating ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Updating...
                </>
              ) : (
                <>
                  <Save className="h-4 w-4 mr-2" />
                  Update Checkpoint
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Diff Comparison Dialog */}
      <Dialog open={showDiffDialog} onOpenChange={setShowDiffDialog}>
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle>Checkpoint Comparison</DialogTitle>
            <DialogDescription>
              Changes between &quot;{selectedCheckpoint?.description || selectedCheckpoint?.name || selectedCheckpoint?.id.slice(0, 8)}&quot; 
              and &quot;{compareCheckpoint?.description || compareCheckpoint?.name || compareCheckpoint?.id.slice(0, 8)}&quot;
            </DialogDescription>
          </DialogHeader>
          
          {diff && (
            <div className="space-y-4 py-4 max-h-[60vh] overflow-y-auto">
              {/* Summary */}
              <div className="grid grid-cols-3 gap-4">
                <Card>
                  <CardContent className="p-3">
                    <div className="text-xs text-muted-foreground">Modified Files</div>
                    <div className="text-2xl font-bold">{diff.modified_files.length}</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-3">
                    <div className="text-xs text-muted-foreground">Added Files</div>
                    <div className="text-2xl font-bold text-green-600">{diff.added_files.length}</div>
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-3">
                    <div className="text-xs text-muted-foreground">Deleted Files</div>
                    <div className="text-2xl font-bold text-red-600">{diff.deleted_files.length}</div>
                  </CardContent>
                </Card>
              </div>
              
              {/* Deltas */}
              <div className="flex items-center justify-center gap-4">
                <Badge variant={diff.token_delta > 0 ? "default" : "secondary"}>
                  {diff.token_delta > 0 ? "+" : ""}{diff.token_delta.toLocaleString()} tokens
                </Badge>
                {/* Message delta not available in CheckpointDiff */}
                {diff.cost_delta !== 0 && (
                  <Badge variant={diff.cost_delta > 0 ? "default" : "secondary"}>
                    {diff.cost_delta > 0 ? "+" : ""}${diff.cost_delta.toFixed(4)}
                  </Badge>
                )}
              </div>
              
              {/* File lists */}
              {diff.modified_files.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                    <FileCode className="h-4 w-4" />
                    Modified Files
                  </h4>
                  <div className="space-y-1">
                    {diff.modified_files.map((file) => (
                      <div key={file.path} className="flex items-center justify-between text-xs p-2 bg-muted rounded">
                        <span className="font-mono">{file.path}</span>
                        <div className="flex items-center gap-2 text-xs">
                          <span className="text-green-600">+{file.additions}</span>
                          <span className="text-red-600">-{file.deletions}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {diff.added_files.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                    <Plus className="h-4 w-4 text-green-600" />
                    Added Files
                  </h4>
                  <div className="space-y-1">
                    {diff.added_files.map((file) => (
                      <div key={file} className="text-xs font-mono text-green-600 p-2 bg-green-50 dark:bg-green-900/20 rounded">
                        + {file}
                      </div>
                    ))}
                  </div>
                </div>
              )}
              
              {diff.deleted_files.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                    <Trash2 className="h-4 w-4 text-red-600" />
                    Deleted Files
                  </h4>
                  <div className="space-y-1">
                    {diff.deleted_files.map((file) => (
                      <div key={file} className="text-xs font-mono text-red-600 p-2 bg-red-50 dark:bg-red-900/20 rounded">
                        - {file}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
          
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setShowDiffDialog(false);
                setDiff(null);
                setCompareCheckpoint(null);
                setSelectedCheckpoint(null);
              }}
            >
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};