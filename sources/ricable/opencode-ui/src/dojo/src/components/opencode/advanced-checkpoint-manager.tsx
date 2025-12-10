/**
 * Advanced Checkpoint Manager - Comprehensive checkpoint system with all advanced features
 * Integrates diff viewer, forking, metadata tracking, and export capabilities
 */

"use client";

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  GitBranch,
  GitFork,
  GitCompare,
  Save,
  RotateCcw,
  Search,
  Filter,
  Download,
  Upload,
  Settings,
  Plus,
  MoreHorizontal,
  Calendar,
  Clock,
  User,
  Zap,
  FileText,
  Hash,
  DollarSign,
  Bookmark,
  CheckCircle,
  AlertCircle,
  Info,
  TrendingUp,
  BarChart3,
  Eye,
  Edit,
  Trash2,
  Share2,
  Maximize2,
  ArrowRight,
  PlayCircle
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger, DropdownMenuSeparator } from '@/components/ui/dropdown-menu';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';
import { 
  Checkpoint, 
  CheckpointDiff, 
  CheckpointStats, 
  CheckpointSearchFilters, 
  CheckpointExportFormat,
  CheckpointForkOptions,
  TimelineNode,
  DiffLine,
  CheckpointMetadata
} from '@/types/opencode';
import { CheckpointDiffViewer } from './checkpoint-diff-viewer';
import { CheckpointForkManager } from './checkpoint-fork-manager';
import { useSessionStore } from '@/lib/session-store';

interface AdvancedCheckpointManagerProps {
  /**
   * Session ID
   */
  sessionId: string;
  /**
   * Whether manager is in sidebar mode
   */
  compact?: boolean;
  /**
   * Callback when checkpoint is selected
   */
  onCheckpointSelect?: (checkpoint: Checkpoint) => void;
  /**
   * Optional className
   */
  className?: string;
}

interface CheckpointItemProps {
  checkpoint: Checkpoint;
  isActive: boolean;
  isSelected: boolean;
  compact?: boolean;
  onSelect: () => void;
  onEdit: () => void;
  onDelete: () => void;
  onRestore: () => void;
  onFork: () => void;
  onCompare: () => void;
  onExport: () => void;
}

const CheckpointItem: React.FC<CheckpointItemProps> = ({
  checkpoint,
  isActive,
  isSelected,
  compact,
  onSelect,
  onEdit,
  onDelete,
  onRestore,
  onFork,
  onCompare,
  onExport
}) => {
  const getTypeIcon = () => {
    switch (checkpoint.type) {
      case 'manual': return <Bookmark className="h-4 w-4" />;
      case 'auto': return <Clock className="h-4 w-4" />;
      case 'milestone': return <CheckCircle className="h-4 w-4" />;
      case 'fork': return <GitFork className="h-4 w-4" />;
      default: return <Bookmark className="h-4 w-4" />;
    }
  };

  const getTypeColor = () => {
    switch (checkpoint.type) {
      case 'manual': return 'text-blue-500 bg-blue-100 dark:bg-blue-900';
      case 'auto': return 'text-gray-500 bg-gray-100 dark:bg-gray-900';
      case 'milestone': return 'text-green-500 bg-green-100 dark:bg-green-900';
      case 'fork': return 'text-purple-500 bg-purple-100 dark:bg-purple-900';
      default: return 'text-gray-500 bg-gray-100 dark:bg-gray-900';
    }
  };

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

  const formatCost = (cost?: number) => {
    if (!cost) return null;
    return cost < 0.01 ? '<$0.01' : `$${cost.toFixed(3)}`;
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="relative"
    >
      <Card className={cn(
        "cursor-pointer transition-all hover:shadow-md",
        isActive && "ring-2 ring-primary bg-primary/5",
        isSelected && "ring-2 ring-blue-500 bg-blue-500/5",
        compact && "p-2"
      )}>
        <CardContent className={cn("p-4", compact && "p-3")}>
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1 min-w-0" onClick={onSelect}>
              {/* Header */}
              <div className="flex items-center gap-2 mb-2">
                <div className={cn(
                  "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center",
                  getTypeColor()
                )}>
                  {getTypeIcon()}
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    {isActive && (
                      <Badge variant="default" className="text-xs">Current</Badge>
                    )}
                    <span className="text-xs font-mono text-muted-foreground">
                      {checkpoint.id.slice(0, 8)}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {formatTime(checkpoint.created_at)}
                    </span>
                  </div>
                  
                  <h4 className="font-medium text-sm truncate mt-1">
                    {checkpoint.name}
                  </h4>
                </div>
              </div>
              
              {/* Description */}
              {checkpoint.description && !compact && (
                <p className="text-xs text-muted-foreground mb-3 line-clamp-2">
                  {checkpoint.description}
                </p>
              )}
              
              {/* Metadata grid */}
              <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
                <div className="flex items-center gap-1">
                  <Hash className="h-3 w-3" />
                  <span>{checkpoint.metadata.totalTokens.toLocaleString()}</span>
                </div>
                
                <div className="flex items-center gap-1">
                  <FileText className="h-3 w-3" />
                  <span>{checkpoint.metadata.fileChanges}</span>
                </div>
                
                {checkpoint.metadata.cost && (
                  <div className="flex items-center gap-1">
                    <DollarSign className="h-3 w-3" />
                    <span>{formatCost(checkpoint.metadata.cost)}</span>
                  </div>
                )}
                
                {checkpoint.metadata.toolsUsed && checkpoint.metadata.toolsUsed.length > 0 && (
                  <div className="flex items-center gap-1">
                    <Zap className="h-3 w-3" />
                    <span>{checkpoint.metadata.toolsUsed.length}</span>
                  </div>
                )}
                
                {checkpoint.metadata.provider && (
                  <div className="flex items-center gap-1 col-span-2">
                    <User className="h-3 w-3" />
                    <span className="truncate">{checkpoint.metadata.provider}</span>
                    {checkpoint.metadata.model && (
                      <span className="text-muted-foreground/60">/ {checkpoint.metadata.model}</span>
                    )}
                  </div>
                )}
              </div>
              
              {/* Type badge */}
              <div className="mt-2">
                <Badge variant="outline" className={cn(
                  "text-xs",
                  checkpoint.type === 'manual' && "border-blue-500 text-blue-700",
                  checkpoint.type === 'auto' && "border-gray-500 text-gray-700",
                  checkpoint.type === 'milestone' && "border-green-500 text-green-700",
                  checkpoint.type === 'fork' && "border-purple-500 text-purple-700"
                )}>
                  {checkpoint.type}
                </Badge>
              </div>
            </div>
            
            {/* Actions */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="h-6 w-6 flex-shrink-0">
                  <MoreHorizontal className="h-3 w-3" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={onSelect}>
                  <PlayCircle className="h-4 w-4 mr-2" />
                  Jump to Checkpoint
                </DropdownMenuItem>
                <DropdownMenuItem onClick={onRestore}>
                  <RotateCcw className="h-4 w-4 mr-2" />
                  Restore Session
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={onFork}>
                  <GitFork className="h-4 w-4 mr-2" />
                  Fork Checkpoint
                </DropdownMenuItem>
                <DropdownMenuItem onClick={onCompare}>
                  <GitCompare className="h-4 w-4 mr-2" />
                  Compare Changes
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={onEdit}>
                  <Edit className="h-4 w-4 mr-2" />
                  Edit Details
                </DropdownMenuItem>
                <DropdownMenuItem onClick={onExport}>
                  <Download className="h-4 w-4 mr-2" />
                  Export
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem 
                  className="text-destructive"
                  onClick={onDelete}
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  Delete
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

export const AdvancedCheckpointManager: React.FC<AdvancedCheckpointManagerProps> = ({
  sessionId,
  compact = false,
  onCheckpointSelect,
  className
}) => {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [selectedCheckpoints, setSelectedCheckpoints] = useState<Set<string>>(new Set());
  const [activeCheckpointId, setActiveCheckpointId] = useState<string | null>(null);
  const [searchFilters, setSearchFilters] = useState<CheckpointSearchFilters>({});
  const [stats, setStats] = useState<CheckpointStats | null>(null);
  
  // Dialog states
  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showEditDialog, setShowEditDialog] = useState(false);
  const [showDiffDialog, setShowDiffDialog] = useState(false);
  const [showForkDialog, setShowForkDialog] = useState(false);
  const [showStatsDialog, setShowStatsDialog] = useState(false);
  
  // Current operations
  const [editingCheckpoint, setEditingCheckpoint] = useState<Checkpoint | null>(null);
  const [forkingCheckpoint, setForkingCheckpoint] = useState<Checkpoint | null>(null);
  const [comparingCheckpoints, setComparingCheckpoints] = useState<[Checkpoint, Checkpoint] | null>(null);
  const [currentDiff, setCurrentDiff] = useState<CheckpointDiff | null>(null);
  
  // Form states
  const [checkpointName, setCheckpointName] = useState('');
  const [checkpointDescription, setCheckpointDescription] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    loadCheckpoints();
    loadStats();
  }, [sessionId, searchFilters]);

  const loadCheckpoints = async () => {
    setIsLoading(true);
    try {
      // Mock data - in real implementation, fetch from API with filters
      const mockCheckpoints: Checkpoint[] = [
        {
          id: 'checkpoint-1',
          session_id: sessionId,
          name: 'Initial Setup',
          description: 'Started working on authentication refactor',
          message_index: 0,
          timestamp: Date.now() - 3600000,
          created_at: Date.now() - 3600000,
          type: 'auto',
          metadata: {
            userPrompt: 'Help me refactor the authentication system',
            model: 'claude-3.7-sonnet',
            provider: 'anthropic',
            totalTokens: 1250,
            inputTokens: 850,
            outputTokens: 400,
            cost: 0.015,
            fileChanges: 3,
            toolsUsed: ['file_read', 'grep'],
            filesModified: ['auth.js', 'middleware.js', 'config.js']
          }
        },
        {
          id: 'checkpoint-2',
          session_id: sessionId,
          name: 'Database Schema Updated',
          description: 'Modified user table to support new auth flow',
          message_index: 5,
          timestamp: Date.now() - 2400000,
          created_at: Date.now() - 2400000,
          type: 'manual',
          metadata: {
            userPrompt: 'Update the database schema for the new authentication system',
            model: 'claude-3.7-sonnet',
            provider: 'anthropic',
            totalTokens: 2100,
            inputTokens: 1200,
            outputTokens: 900,
            cost: 0.042,
            fileChanges: 5,
            toolsUsed: ['file_write', 'bash'],
            filesModified: ['schema.sql', 'migrations/001_update_users.sql', 'models/user.js']
          }
        },
        {
          id: 'checkpoint-3',
          session_id: sessionId,
          name: 'API Endpoints Complete',
          description: 'Implemented all authentication endpoints with JWT support',
          message_index: 12,
          timestamp: Date.now() - 1200000,
          created_at: Date.now() - 1200000,
          type: 'milestone',
          metadata: {
            userPrompt: 'Create the REST API endpoints for authentication',
            model: 'claude-3.7-sonnet',
            provider: 'anthropic',
            totalTokens: 3200,
            inputTokens: 1800,
            outputTokens: 1400,
            cost: 0.076,
            fileChanges: 8,
            toolsUsed: ['file_write', 'bash', 'grep'],
            filesModified: ['routes/auth.js', 'controllers/auth.js', 'middleware/jwt.js', 'tests/auth.test.js']
          }
        }
      ];
      
      setCheckpoints(mockCheckpoints);
      setActiveCheckpointId(mockCheckpoints[mockCheckpoints.length - 1]?.id || null);
    } catch (error) {
      console.error('Failed to load checkpoints:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const loadStats = async () => {
    // Mock stats data
    const mockStats: CheckpointStats = {
      total_checkpoints: 15,
      manual_checkpoints: 8,
      auto_checkpoints: 5,
      milestone_checkpoints: 2,
      fork_checkpoints: 0,
      total_size_bytes: 1024 * 1024 * 12.5,
      avg_files_per_checkpoint: 4.2,
      most_active_day: '2024-01-15',
      cost_breakdown: {
        total: 0.245,
        by_provider: {
          'anthropic': 0.245
        },
        by_model: {
          'claude-3.7-sonnet': 0.245
        }
      }
    };
    setStats(mockStats);
  };

  const handleCreateCheckpoint = async () => {
    if (!checkpointName.trim()) return;
    
    setIsLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      await loadCheckpoints();
      setShowCreateDialog(false);
      setCheckpointName('');
      setCheckpointDescription('');
    } catch (error) {
      console.error('Failed to create checkpoint:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleForkCheckpoint = (checkpoint: Checkpoint) => {
    setForkingCheckpoint(checkpoint);
    setShowForkDialog(true);
  };

  const handleCompareCheckpoints = async (checkpoint: Checkpoint) => {
    if (selectedCheckpoints.size === 0) {
      setSelectedCheckpoints(new Set([checkpoint.id]));
      return;
    }
    
    if (selectedCheckpoints.size === 1) {
      const firstCheckpoint = checkpoints.find(cp => selectedCheckpoints.has(cp.id));
      if (firstCheckpoint) {
        setComparingCheckpoints([firstCheckpoint, checkpoint]);
        
        // Generate mock diff data
        const mockDiff: CheckpointDiff = {
          from_checkpoint: firstCheckpoint,
          to_checkpoint: checkpoint,
          modified_files: [
            {
              path: 'auth.js',
              old_content: 'const auth = require("./old-auth");',
              new_content: 'const auth = require("./new-auth");',
              additions: 15,
              deletions: 8,
              changes: []
            }
          ],
          added_files: ['middleware/jwt.js'],
          deleted_files: ['old-auth.js'],
          token_delta: checkpoint.metadata.totalTokens - firstCheckpoint.metadata.totalTokens,
          cost_delta: (checkpoint.metadata.cost || 0) - (firstCheckpoint.metadata.cost || 0),
          total_changes: 23
        };
        
        setCurrentDiff(mockDiff);
        setShowDiffDialog(true);
        setSelectedCheckpoints(new Set());
      }
    }
  };

  const filteredCheckpoints = checkpoints.filter(checkpoint => {
    if (searchFilters.keyword && !checkpoint.name.toLowerCase().includes(searchFilters.keyword.toLowerCase())) {
      return false;
    }
    if (searchFilters.type && searchFilters.type.length > 0 && !searchFilters.type.includes(checkpoint.type)) {
      return false;
    }
    if (searchFilters.provider && checkpoint.metadata.provider !== searchFilters.provider) {
      return false;
    }
    return true;
  });

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <GitBranch className="h-5 w-5" />
            <h3 className="font-medium">
              {compact ? 'Checkpoints' : 'Advanced Checkpoint Manager'}
            </h3>
            {stats && (
              <Badge variant="outline" className="text-xs">
                {stats.total_checkpoints} total
              </Badge>
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            {!compact && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() => setShowStatsDialog(true)}
                    >
                      <BarChart3 className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>View Statistics</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
            
            <Button
              variant="default"
              size={compact ? "sm" : "default"}
              onClick={() => setShowCreateDialog(true)}
            >
              <Plus className="h-4 w-4 mr-1" />
              {compact ? 'New' : 'Create Checkpoint'}
            </Button>
          </div>
        </div>
        
        {!compact && (
          <div className="flex items-center space-x-2">
            <Search className="h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search checkpoints..."
              value={searchFilters.keyword || ''}
              onChange={(e) => setSearchFilters(prev => ({...prev, keyword: e.target.value}))}
              className="flex-1"
            />
            <Select
              value={searchFilters.type?.[0] || 'all'}
              onValueChange={(value) => setSearchFilters(prev => ({
                ...prev, 
                type: value === 'all' ? undefined : [value as any]
              }))}
            >
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="manual">Manual</SelectItem>
                <SelectItem value="auto">Auto</SelectItem>
                <SelectItem value="milestone">Milestone</SelectItem>
                <SelectItem value="fork">Fork</SelectItem>
              </SelectContent>
            </Select>
          </div>
        )}
      </div>

      {/* Checkpoint list */}
      <div className="flex-1 overflow-hidden">
        <ScrollArea className="h-full">
          <div className="p-4 space-y-4">
            {filteredCheckpoints.length === 0 ? (
              <div className="text-center py-8">
                <Clock className="h-8 w-8 text-muted-foreground mx-auto mb-3" />
                <p className="text-sm text-muted-foreground mb-2">
                  {searchFilters.keyword ? 'No checkpoints match your search' : 'No checkpoints yet'}
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setShowCreateDialog(true)}
                >
                  <Plus className="h-4 w-4 mr-2" />
                  Create First Checkpoint
                </Button>
              </div>
            ) : (
              <AnimatePresence>
                {filteredCheckpoints.map((checkpoint, index) => (
                  <CheckpointItem
                    key={checkpoint.id}
                    checkpoint={checkpoint}
                    isActive={activeCheckpointId === checkpoint.id}
                    isSelected={selectedCheckpoints.has(checkpoint.id)}
                    compact={compact}
                    onSelect={() => {
                      setActiveCheckpointId(checkpoint.id);
                      onCheckpointSelect?.(checkpoint);
                    }}
                    onEdit={() => {
                      setEditingCheckpoint(checkpoint);
                      setCheckpointName(checkpoint.name);
                      setCheckpointDescription(checkpoint.description || '');
                      setShowEditDialog(true);
                    }}
                    onDelete={() => {
                      if (confirm(`Delete checkpoint "${checkpoint.name}"?`)) {
                        setCheckpoints(prev => prev.filter(cp => cp.id !== checkpoint.id));
                      }
                    }}
                    onRestore={() => {
                      if (confirm(`Restore to checkpoint "${checkpoint.name}"?`)) {
                        setActiveCheckpointId(checkpoint.id);
                        onCheckpointSelect?.(checkpoint);
                      }
                    }}
                    onFork={() => handleForkCheckpoint(checkpoint)}
                    onCompare={() => handleCompareCheckpoints(checkpoint)}
                    onExport={() => {
                      // Trigger export functionality
                      console.log('Export checkpoint:', checkpoint.id);
                    }}
                  />
                ))}
              </AnimatePresence>
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Dialogs */}
      {/* Create Dialog */}
      <Dialog open={showCreateDialog} onOpenChange={setShowCreateDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create Checkpoint</DialogTitle>
            <DialogDescription>
              Save the current state of your session
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4">
            <div>
              <Label htmlFor="name">Name *</Label>
              <Input
                id="name"
                value={checkpointName}
                onChange={(e) => setCheckpointName(e.target.value)}
                placeholder="Enter checkpoint name..."
              />
            </div>
            <div>
              <Label htmlFor="description">Description</Label>
              <Textarea
                id="description"
                value={checkpointDescription}
                onChange={(e) => setCheckpointDescription(e.target.value)}
                placeholder="Describe what was accomplished..."
                rows={3}
              />
            </div>
          </div>
          
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowCreateDialog(false)}>
              Cancel
            </Button>
            <Button
              onClick={handleCreateCheckpoint}
              disabled={!checkpointName.trim() || isLoading}
            >
              {isLoading ? 'Creating...' : 'Create Checkpoint'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Diff Dialog */}
      {showDiffDialog && currentDiff && (
        <Dialog open={showDiffDialog} onOpenChange={setShowDiffDialog}>
          <DialogContent className="max-w-[95vw] max-h-[95vh] overflow-hidden">
            <CheckpointDiffViewer
              diff={currentDiff}
              fullscreen={false}
              onExport={(format) => console.log('Export diff:', format)}
            />
          </DialogContent>
        </Dialog>
      )}

      {/* Fork Dialog */}
      {showForkDialog && forkingCheckpoint && (
        <CheckpointForkManager
          sourceCheckpoint={forkingCheckpoint}
          timeline={[]} // Pass timeline data
          isDialog={true}
          onForkCreated={(forkId, options) => {
            console.log('Fork created:', forkId, options);
            setShowForkDialog(false);
            setForkingCheckpoint(null);
            loadCheckpoints();
          }}
          onCancel={() => {
            setShowForkDialog(false);
            setForkingCheckpoint(null);
          }}
        />
      )}

      {/* Stats Dialog */}
      {showStatsDialog && stats && (
        <Dialog open={showStatsDialog} onOpenChange={setShowStatsDialog}>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>Checkpoint Statistics</DialogTitle>
            </DialogHeader>
            
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Total Checkpoints</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{stats.total_checkpoints}</div>
                  </CardContent>
                </Card>
                
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">Total Cost</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">${stats.cost_breakdown.total.toFixed(3)}</div>
                  </CardContent>
                </Card>
              </div>
              
              <div className="space-y-2">
                <h4 className="font-medium">Checkpoint Types</h4>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Manual</span>
                    <div className="flex items-center space-x-2">
                      <Progress value={(stats.manual_checkpoints / stats.total_checkpoints) * 100} className="w-24" />
                      <span className="text-sm">{stats.manual_checkpoints}</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Auto</span>
                    <div className="flex items-center space-x-2">
                      <Progress value={(stats.auto_checkpoints / stats.total_checkpoints) * 100} className="w-24" />
                      <span className="text-sm">{stats.auto_checkpoints}</span>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Milestone</span>
                    <div className="flex items-center space-x-2">
                      <Progress value={(stats.milestone_checkpoints / stats.total_checkpoints) * 100} className="w-24" />
                      <span className="text-sm">{stats.milestone_checkpoints}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
};