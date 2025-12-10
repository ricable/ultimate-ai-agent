/**
 * Checkpoint Fork Manager - Advanced forking and branching capabilities
 * Provides visual interface for creating and managing checkpoint forks
 */

"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  GitBranch,
  GitFork,
  GitMerge,
  Plus,
  Trash2,
  Edit,
  Copy,
  Share2,
  Download,
  Upload,
  Settings,
  AlertCircle,
  CheckCircle,
  Clock,
  User,
  FileText,
  Zap,
  Hash,
  ArrowRight,
  Maximize2,
  Eye,
  EyeOff
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { cn } from '@/lib/utils';
import { Checkpoint, CheckpointForkOptions, TimelineNode } from '@/types/opencode';

interface CheckpointForkManagerProps {
  /**
   * The checkpoint to fork from
   */
  sourceCheckpoint: Checkpoint;
  /**
   * Session timeline data
   */
  timeline: TimelineNode[];
  /**
   * Whether the manager is in dialog mode
   */
  isDialog?: boolean;
  /**
   * Callback when fork is created
   */
  onForkCreated?: (forkId: string, options: CheckpointForkOptions) => void;
  /**
   * Callback when fork is cancelled
   */
  onCancel?: () => void;
  /**
   * Optional className
   */
  className?: string;
}

interface ForkPreview {
  estimatedSize: number;
  fileCount: number;
  messageCount: number;
  dependencies: string[];
  warnings: string[];
}

interface BranchInfo {
  id: string;
  name: string;
  description?: string;
  created_at: number;
  checkpoint_count: number;
  last_activity: number;
  is_active: boolean;
  parent_branch?: string;
}

const BranchVisualizer: React.FC<{
  branches: BranchInfo[];
  currentBranch: string;
  onBranchSelect: (branchId: string) => void;
}> = ({ branches, currentBranch, onBranchSelect }) => {
  const [expandedBranches, setExpandedBranches] = useState<Set<string>>(new Set([currentBranch]));

  const toggleBranch = (branchId: string) => {
    const newExpanded = new Set(expandedBranches);
    if (newExpanded.has(branchId)) {
      newExpanded.delete(branchId);
    } else {
      newExpanded.add(branchId);
    }
    setExpandedBranches(newExpanded);
  };

  const getBranchDepth = (branch: BranchInfo): number => {
    if (!branch.parent_branch) return 0;
    const parent = branches.find(b => b.id === branch.parent_branch);
    return parent ? getBranchDepth(parent) + 1 : 0;
  };

  return (
    <div className="space-y-2">
      {branches.map((branch) => {
        const depth = getBranchDepth(branch);
        const isExpanded = expandedBranches.has(branch.id);
        const isCurrent = branch.id === currentBranch;
        const hasChildren = branches.some(b => b.parent_branch === branch.id);

        return (
          <motion.div
            key={branch.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className={cn(
              "relative cursor-pointer",
              depth > 0 && "ml-6"
            )}
            style={{ paddingLeft: `${depth * 24}px` }}
          >
            {/* Connection line */}
            {depth > 0 && (
              <div className="absolute left-0 top-0 w-6 h-6 border-l-2 border-b-2 border-muted-foreground/30 rounded-bl" />
            )}

            <Card 
              className={cn(
                "transition-all hover:shadow-md",
                isCurrent && "border-primary ring-2 ring-primary/20",
                !branch.is_active && "opacity-60"
              )}
              onClick={() => onBranchSelect(branch.id)}
            >
              <CardContent className="p-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <GitBranch className={cn(
                      "h-4 w-4",
                      isCurrent ? "text-primary" : "text-muted-foreground"
                    )} />
                    <div>
                      <div className="font-medium text-sm">{branch.name}</div>
                      {branch.description && (
                        <div className="text-xs text-muted-foreground">{branch.description}</div>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    {isCurrent && (
                      <Badge variant="default" className="text-xs">Current</Badge>
                    )}
                    {!branch.is_active && (
                      <Badge variant="outline" className="text-xs">Archived</Badge>
                    )}
                    <div className="text-xs text-muted-foreground">
                      {branch.checkpoint_count} checkpoints
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        );
      })}
    </div>
  );
};

export const CheckpointForkManager: React.FC<CheckpointForkManagerProps> = ({
  sourceCheckpoint,
  timeline,
  isDialog = true,
  onForkCreated,
  onCancel,
  className
}) => {
  const [forkOptions, setForkOptions] = useState<CheckpointForkOptions>({
    name: `Fork from ${sourceCheckpoint.name || sourceCheckpoint.id.slice(0, 8)}`,
    description: '',
    preserve_messages: true,
    preserve_files: true,
    create_branch: true
  });
  
  const [preview, setPreview] = useState<ForkPreview | null>(null);
  const [isCreating, setIsCreating] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [branches, setBranches] = useState<BranchInfo[]>([]);
  const [selectedBranch, setSelectedBranch] = useState<string>('main');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Load branch information
    loadBranches();
    // Generate preview
    generatePreview();
  }, [sourceCheckpoint, forkOptions]);

  const loadBranches = async () => {
    // Mock branch data - in real implementation, fetch from API
    const mockBranches: BranchInfo[] = [
      {
        id: 'main',
        name: 'main',
        description: 'Main development branch',
        created_at: Date.now() - 86400000 * 30,
        checkpoint_count: 15,
        last_activity: Date.now() - 3600000,
        is_active: true
      },
      {
        id: 'feature-auth',
        name: 'feature/authentication',
        description: 'User authentication system',
        created_at: Date.now() - 86400000 * 7,
        checkpoint_count: 8,
        last_activity: Date.now() - 86400000,
        is_active: true,
        parent_branch: 'main'
      },
      {
        id: 'experimental',
        name: 'experimental',
        description: 'Experimental features and prototypes',
        created_at: Date.now() - 86400000 * 14,
        checkpoint_count: 3,
        last_activity: Date.now() - 86400000 * 3,
        is_active: false,
        parent_branch: 'main'
      }
    ];
    setBranches(mockBranches);
  };

  const generatePreview = async () => {
    // Calculate preview data based on options
    const estimatedSize = 1024 * 1024 * 2.5; // 2.5MB estimated
    const fileCount = sourceCheckpoint.metadata.filesModified?.length || 0;
    const messageCount = sourceCheckpoint.message_index + 1;
    
    const dependencies: string[] = [];
    if (sourceCheckpoint.parent_checkpoint_id) {
      dependencies.push(sourceCheckpoint.parent_checkpoint_id);
    }
    
    const warnings: string[] = [];
    if (!forkOptions.preserve_files) {
      warnings.push('File history will not be preserved');
    }
    if (!forkOptions.preserve_messages) {
      warnings.push('Message history will be truncated');
    }
    if (estimatedSize > 1024 * 1024 * 10) {
      warnings.push('Large fork size may affect performance');
    }

    setPreview({
      estimatedSize,
      fileCount,
      messageCount,
      dependencies,
      warnings
    });
  };

  const handleCreateFork = async () => {
    if (!forkOptions.name?.trim()) {
      setError('Fork name is required');
      return;
    }

    setIsCreating(true);
    setError(null);

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const forkId = `fork-${Date.now()}`;
      onForkCreated?.(forkId, forkOptions);
    } catch (err) {
      setError('Failed to create fork');
    } finally {
      setIsCreating(false);
    }
  };

  const formatFileSize = (bytes: number) => {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  const content = (
    <div className="space-y-6">
      {/* Source checkpoint info */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <GitFork className="h-5 w-5" />
            <span>Fork Source</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-sm text-muted-foreground">Checkpoint</div>
              <div className="font-mono text-sm">{sourceCheckpoint.id.slice(0, 8)}</div>
              <div className="font-medium">{sourceCheckpoint.name || 'Unnamed checkpoint'}</div>
              {sourceCheckpoint.description && (
                <div className="text-sm text-muted-foreground mt-1">{sourceCheckpoint.description}</div>
              )}
            </div>
            <div className="space-y-2">
              <div className="flex items-center space-x-2 text-sm">
                <Clock className="h-4 w-4" />
                <span>{formatTimestamp(sourceCheckpoint.created_at)}</span>
              </div>
              <div className="flex items-center space-x-2 text-sm">
                <Hash className="h-4 w-4" />
                <span>{sourceCheckpoint.metadata.totalTokens.toLocaleString()} tokens</span>
              </div>
              <div className="flex items-center space-x-2 text-sm">
                <FileText className="h-4 w-4" />
                <span>{sourceCheckpoint.metadata.fileChanges} files</span>
              </div>
              {sourceCheckpoint.metadata.toolsUsed && (
                <div className="flex items-center space-x-2 text-sm">
                  <Zap className="h-4 w-4" />
                  <span>{sourceCheckpoint.metadata.toolsUsed.length} tools used</span>
                </div>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Fork configuration */}
      <Card>
        <CardHeader>
          <CardTitle>Fork Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label htmlFor="fork-name">Fork Name *</Label>
            <Input
              id="fork-name"
              value={forkOptions.name}
              onChange={(e) => setForkOptions(prev => ({...prev, name: e.target.value}))}
              placeholder="Enter fork name..."
            />
          </div>

          <div>
            <Label htmlFor="fork-description">Description</Label>
            <Textarea
              id="fork-description"
              value={forkOptions.description}
              onChange={(e) => setForkOptions(prev => ({...prev, description: e.target.value}))}
              placeholder="Describe the purpose of this fork..."
              rows={3}
            />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="preserve-messages">Preserve Message History</Label>
              <div className="text-sm text-muted-foreground">Include all messages up to this checkpoint</div>
            </div>
            <Switch
              id="preserve-messages"
              checked={forkOptions.preserve_messages}
              onCheckedChange={(checked) => setForkOptions(prev => ({...prev, preserve_messages: checked}))}
            />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="preserve-files">Preserve File History</Label>
              <div className="text-sm text-muted-foreground">Include file snapshots and diffs</div>
            </div>
            <Switch
              id="preserve-files"
              checked={forkOptions.preserve_files}
              onCheckedChange={(checked) => setForkOptions(prev => ({...prev, preserve_files: checked}))}
            />
          </div>

          <div className="flex items-center justify-between">
            <div>
              <Label htmlFor="create-branch">Create New Branch</Label>
              <div className="text-sm text-muted-foreground">Start a new branch for this fork</div>
            </div>
            <Switch
              id="create-branch"
              checked={forkOptions.create_branch}
              onCheckedChange={(checked) => setForkOptions(prev => ({...prev, create_branch: checked}))}
            />
          </div>

          <Button
            variant="outline"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="w-full"
          >
            {showAdvanced ? 'Hide' : 'Show'} Advanced Options
            {showAdvanced ? <EyeOff className="h-4 w-4 ml-2" /> : <Eye className="h-4 w-4 ml-2" />}
          </Button>

          <AnimatePresence>
            {showAdvanced && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="space-y-4 border-t pt-4"
              >
                <div>
                  <Label>Target Branch</Label>
                  <Select value={selectedBranch} onValueChange={setSelectedBranch}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {branches.map(branch => (
                        <SelectItem key={branch.id} value={branch.id}>
                          {branch.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <Label>Branch Visualization</Label>
                  <div className="mt-2 p-4 border rounded-lg max-h-60 overflow-y-auto">
                    <BranchVisualizer
                      branches={branches}
                      currentBranch={selectedBranch}
                      onBranchSelect={setSelectedBranch}
                    />
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </CardContent>
      </Card>

      {/* Preview */}
      {preview && (
        <Card>
          <CardHeader>
            <CardTitle>Fork Preview</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <div className="text-sm text-muted-foreground">Estimated Size</div>
                <div className="text-lg font-medium">{formatFileSize(preview.estimatedSize)}</div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground">Files Included</div>
                <div className="text-lg font-medium">{preview.fileCount}</div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground">Messages</div>
                <div className="text-lg font-medium">{preview.messageCount}</div>
              </div>
              <div>
                <div className="text-sm text-muted-foreground">Dependencies</div>
                <div className="text-lg font-medium">{preview.dependencies.length}</div>
              </div>
            </div>

            {preview.warnings.length > 0 && (
              <div className="border-l-4 border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20 p-3">
                <div className="flex items-start space-x-2">
                  <AlertCircle className="h-4 w-4 text-yellow-600 mt-0.5" />
                  <div>
                    <div className="text-sm font-medium text-yellow-800 dark:text-yellow-200">Warnings</div>
                    <ul className="text-sm text-yellow-700 dark:text-yellow-300 space-y-1 mt-1">
                      {preview.warnings.map((warning, index) => (
                        <li key={index}>• {warning}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {error && (
        <div className="border-l-4 border-red-500 bg-red-50 dark:bg-red-900/20 p-3">
          <div className="flex items-center space-x-2">
            <AlertCircle className="h-4 w-4 text-red-600" />
            <div className="text-sm text-red-800 dark:text-red-200">{error}</div>
          </div>
        </div>
      )}
    </div>
  );

  if (!isDialog) {
    return (
      <div className={cn("space-y-6", className)}>
        {content}
        <div className="flex items-center justify-end space-x-2">
          {onCancel && (
            <Button variant="outline" onClick={onCancel}>
              Cancel
            </Button>
          )}
          <Button
            onClick={handleCreateFork}
            disabled={isCreating || !forkOptions.name?.trim()}
          >
            {isCreating ? (
              <>
                <Clock className="h-4 w-4 mr-2 animate-spin" />
                Creating Fork...
              </>
            ) : (
              <>
                <GitFork className="h-4 w-4 mr-2" />
                Create Fork
              </>
            )}
          </Button>
        </div>
      </div>
    );
  }

  return (
    <Dialog open={true} onOpenChange={() => onCancel?.()}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center space-x-2">
            <GitFork className="h-5 w-5" />
            <span>Create Checkpoint Fork</span>
          </DialogTitle>
          <DialogDescription>
            Create a new branch from this checkpoint to explore different paths
          </DialogDescription>
        </DialogHeader>

        {content}

        <DialogFooter>
          <Button variant="outline" onClick={onCancel} disabled={isCreating}>
            Cancel
          </Button>
          <Button
            onClick={handleCreateFork}
            disabled={isCreating || !forkOptions.name?.trim()}
          >
            {isCreating ? (
              <>
                <Clock className="h-4 w-4 mr-2 animate-spin" />
                Creating...
              </>
            ) : (
              <>
                <GitFork className="h-4 w-4 mr-2" />
                Create Fork
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};