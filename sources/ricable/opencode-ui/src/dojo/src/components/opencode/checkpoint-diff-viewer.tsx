/**
 * Checkpoint Diff Viewer - Advanced diff comparison with syntax highlighting
 * Provides side-by-side and unified diff views for checkpoint comparisons
 */

"use client";

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ChevronLeft,
  ChevronRight,
  GitCompare,
  FileText,
  FolderOpen,
  Search,
  Filter,
  Download,
  Copy,
  Maximize2,
  Minimize2,
  Settings,
  RotateCcw,
  Eye,
  EyeOff,
  Split,
  Layers,
  Hash,
  Clock,
  User,
  Zap
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { cn } from '@/lib/utils';
import { CheckpointDiff, FileDiff, DiffLine, Checkpoint } from '@/types/opencode';

interface CheckpointDiffViewerProps {
  /**
   * Diff data to display
   */
  diff: CheckpointDiff;
  /**
   * Whether the viewer is in fullscreen mode
   */
  fullscreen?: boolean;
  /**
   * Callback when fullscreen state changes
   */
  onFullscreenChange?: (fullscreen: boolean) => void;
  /**
   * Callback when export is requested
   */
  onExport?: (format: string) => void;
  /**
   * Optional className
   */
  className?: string;
}

interface DiffViewSettings {
  viewMode: 'side-by-side' | 'unified';
  showLineNumbers: boolean;
  showWhitespace: boolean;
  wrapLines: boolean;
  contextLines: number;
  highlightSyntax: boolean;
  theme: 'light' | 'dark' | 'auto';
}

interface FileDiffViewProps {
  fileDiff: FileDiff;
  settings: DiffViewSettings;
  searchTerm?: string;
}

const DiffLineComponent: React.FC<{
  line: DiffLine;
  lineNumber?: number;
  settings: DiffViewSettings;
  isHighlighted?: boolean;
}> = ({ line, lineNumber, settings, isHighlighted }) => {
  const getLineClassName = () => {
    const base = "font-mono text-sm px-2 py-0.5 border-l-2 transition-colors";
    switch (line.type) {
      case 'added':
        return cn(base, "bg-green-50 border-green-500 text-green-900 dark:bg-green-900/20 dark:text-green-100");
      case 'removed':
        return cn(base, "bg-red-50 border-red-500 text-red-900 dark:bg-red-900/20 dark:text-red-100");
      case 'context':
        return cn(base, "bg-transparent border-transparent text-muted-foreground");
      default:
        return cn(base, "bg-transparent border-transparent");
    }
  };

  const getLinePrefix = () => {
    switch (line.type) {
      case 'added': return '+';
      case 'removed': return '-';
      case 'context': return ' ';
      default: return ' ';
    }
  };

  return (
    <div className={cn(getLineClassName(), isHighlighted && "ring-2 ring-blue-500")}>
      <div className="flex items-center space-x-2">
        {settings.showLineNumbers && (
          <span className="text-xs text-muted-foreground w-8 text-right">
            {lineNumber}
          </span>
        )}
        <span className="text-muted-foreground w-4 text-center">
          {getLinePrefix()}
        </span>
        <span className={cn(
          "flex-1",
          settings.wrapLines ? "whitespace-pre-wrap" : "whitespace-pre",
          settings.showWhitespace && "whitespace-visible"
        )}>
          {line.content}
        </span>
      </div>
    </div>
  );
};

const FileDiffView: React.FC<FileDiffViewProps> = ({ fileDiff, settings, searchTerm }) => {
  const [selectedHunk, setSelectedHunk] = useState<number | null>(null);
  const [highlightedLines, setHighlightedLines] = useState<Set<number>>(new Set());

  useEffect(() => {
    if (searchTerm) {
      const matches = new Set<number>();
      fileDiff.changes.forEach((hunk, hunkIndex) => {
        hunk.lines.forEach((line, lineIndex) => {
          if (line.content.toLowerCase().includes(searchTerm.toLowerCase())) {
            matches.add(hunkIndex * 1000 + lineIndex); // Simple encoding for line identification
          }
        });
      });
      setHighlightedLines(matches);
    } else {
      setHighlightedLines(new Set());
    }
  }, [searchTerm, fileDiff]);

  const renderSideBySide = () => (
    <div className="grid grid-cols-2 gap-1 h-full">
      {/* Old content */}
      <div className="border-r">
        <div className="bg-muted/50 px-2 py-1 text-xs font-medium border-b">
          Original
        </div>
        <ScrollArea className="h-[600px]">
          <div className="p-2">
            {fileDiff.changes.map((hunk, hunkIndex) => (
              <div key={hunkIndex} className="mb-4">
                <div className="text-xs text-muted-foreground mb-1 px-2">
                  @@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@
                </div>
                {hunk.lines
                  .filter(line => line.type !== 'added')
                  .map((line, lineIndex) => (
                    <DiffLineComponent
                      key={`old-${hunkIndex}-${lineIndex}`}
                      line={line}
                      lineNumber={line.old_line_number}
                      settings={settings}
                      isHighlighted={highlightedLines.has(hunkIndex * 1000 + lineIndex)}
                    />
                  ))}
              </div>
            ))}
          </div>
        </ScrollArea>
      </div>

      {/* New content */}
      <div>
        <div className="bg-muted/50 px-2 py-1 text-xs font-medium border-b">
          Modified
        </div>
        <ScrollArea className="h-[600px]">
          <div className="p-2">
            {fileDiff.changes.map((hunk, hunkIndex) => (
              <div key={hunkIndex} className="mb-4">
                <div className="text-xs text-muted-foreground mb-1 px-2">
                  @@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@
                </div>
                {hunk.lines
                  .filter(line => line.type !== 'removed')
                  .map((line, lineIndex) => (
                    <DiffLineComponent
                      key={`new-${hunkIndex}-${lineIndex}`}
                      line={line}
                      lineNumber={line.new_line_number}
                      settings={settings}
                      isHighlighted={highlightedLines.has(hunkIndex * 1000 + lineIndex)}
                    />
                  ))}
              </div>
            ))}
          </div>
        </ScrollArea>
      </div>
    </div>
  );

  const renderUnified = () => (
    <ScrollArea className="h-[600px]">
      <div className="p-2">
        {fileDiff.changes.map((hunk, hunkIndex) => (
          <div key={hunkIndex} className="mb-4">
            <div className="text-xs text-muted-foreground mb-1 px-2 bg-muted/50 py-1 rounded">
              @@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@
            </div>
            {hunk.lines.map((line, lineIndex) => (
              <DiffLineComponent
                key={`unified-${hunkIndex}-${lineIndex}`}
                line={line}
                lineNumber={line.new_line_number || line.old_line_number}
                settings={settings}
                isHighlighted={highlightedLines.has(hunkIndex * 1000 + lineIndex)}
              />
            ))}
          </div>
        ))}
      </div>
    </ScrollArea>
  );

  return (
    <div className="border rounded-lg overflow-hidden">
      {/* File header */}
      <div className="bg-muted/50 px-3 py-2 border-b flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <FileText className="h-4 w-4" />
          <span className="font-mono text-sm">{fileDiff.path}</span>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant="outline" className="text-xs">
            <span className="text-green-600">+{fileDiff.additions}</span>
            <span className="mx-1">/</span>
            <span className="text-red-600">-{fileDiff.deletions}</span>
          </Badge>
        </div>
      </div>

      {/* Diff content */}
      {settings.viewMode === 'side-by-side' ? renderSideBySide() : renderUnified()}
    </div>
  );
};

export const CheckpointDiffViewer: React.FC<CheckpointDiffViewerProps> = ({
  diff,
  fullscreen = false,
  onFullscreenChange,
  onExport,
  className
}) => {
  const [settings, setSettings] = useState<DiffViewSettings>({
    viewMode: 'side-by-side',
    showLineNumbers: true,
    showWhitespace: false,
    wrapLines: false,
    contextLines: 3,
    highlightSyntax: true,
    theme: 'auto'
  });
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [fileFilter, setFileFilter] = useState<'all' | 'modified' | 'added' | 'deleted'>('all');

  const filteredFiles = React.useMemo(() => {
    let files: Array<{type: 'modified' | 'added' | 'deleted', path: string, diff?: FileDiff}> = [];
    
    if (fileFilter === 'all' || fileFilter === 'modified') {
      files = files.concat(diff.modified_files.map(f => ({type: 'modified' as const, path: f.path, diff: f})));
    }
    if (fileFilter === 'all' || fileFilter === 'added') {
      files = files.concat(diff.added_files.map(f => ({type: 'added' as const, path: f})));
    }
    if (fileFilter === 'all' || fileFilter === 'deleted') {
      files = files.concat(diff.deleted_files.map(f => ({type: 'deleted' as const, path: f})));
    }

    if (searchTerm) {
      files = files.filter(f => f.path.toLowerCase().includes(searchTerm.toLowerCase()));
    }

    return files;
  }, [diff, fileFilter, searchTerm]);

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleString();
  };

  const getFileTypeIcon = (type: 'modified' | 'added' | 'deleted') => {
    switch (type) {
      case 'modified': return '📝';
      case 'added': return '➕';
      case 'deleted': return '❌';
    }
  };

  const copyToClipboard = (content: string) => {
    navigator.clipboard.writeText(content);
  };

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* Header */}
      <div className="border-b p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <GitCompare className="h-5 w-5" />
            <div>
              <h3 className="font-semibold">Checkpoint Comparison</h3>
              <p className="text-sm text-muted-foreground">
                {diff.from_checkpoint.name || diff.from_checkpoint.id.slice(0, 8)} → {diff.to_checkpoint.name || diff.to_checkpoint.id.slice(0, 8)}
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => setShowSettings(true)}
                  >
                    <Settings className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Diff Settings</TooltipContent>
              </Tooltip>
            </TooltipProvider>
            
            {onExport && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() => onExport('pdf')}
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Export Diff</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
            
            {onFullscreenChange && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      variant="outline"
                      size="icon"
                      onClick={() => onFullscreenChange(!fullscreen)}
                    >
                      {fullscreen ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>{fullscreen ? 'Exit Fullscreen' : 'Fullscreen'}</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
          </div>
        </div>

        {/* Diff summary */}
        <div className="grid grid-cols-4 gap-4 mb-4">
          <Card>
            <CardContent className="p-3">
              <div className="text-sm text-muted-foreground">Modified Files</div>
              <div className="text-2xl font-bold">{diff.modified_files.length}</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-3">
              <div className="text-sm text-muted-foreground">Added Files</div>
              <div className="text-2xl font-bold text-green-600">{diff.added_files.length}</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-3">
              <div className="text-sm text-muted-foreground">Deleted Files</div>
              <div className="text-2xl font-bold text-red-600">{diff.deleted_files.length}</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-3">
              <div className="text-sm text-muted-foreground">Token Delta</div>
              <div className={cn("text-2xl font-bold", diff.token_delta >= 0 ? "text-green-600" : "text-red-600")}>
                {diff.token_delta >= 0 ? '+' : ''}{diff.token_delta.toLocaleString()}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Controls */}
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <Search className="h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search files..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-64"
            />
          </div>
          
          <Select value={fileFilter} onValueChange={(value: any) => setFileFilter(value)}>
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Files</SelectItem>
              <SelectItem value="modified">Modified</SelectItem>
              <SelectItem value="added">Added</SelectItem>
              <SelectItem value="deleted">Deleted</SelectItem>
            </SelectContent>
          </Select>

          <Tabs value={settings.viewMode} onValueChange={(value: any) => setSettings(prev => ({...prev, viewMode: value}))}>
            <TabsList>
              <TabsTrigger value="side-by-side">
                <Split className="h-4 w-4 mr-1" />
                Side-by-Side
              </TabsTrigger>
              <TabsTrigger value="unified">
                <Layers className="h-4 w-4 mr-1" />
                Unified
              </TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
      </div>

      {/* Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* File list sidebar */}
        <div className="w-80 border-r">
          <div className="p-3 border-b">
            <h4 className="font-medium text-sm">Changed Files ({filteredFiles.length})</h4>
          </div>
          <ScrollArea className="h-full">
            <div className="p-2 space-y-1">
              {filteredFiles.map((file, index) => (
                <motion.div
                  key={`${file.type}-${file.path}`}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.2, delay: index * 0.02 }}
                  className={cn(
                    "flex items-center space-x-2 p-2 rounded cursor-pointer hover:bg-muted/50 transition-colors",
                    selectedFile === file.path && "bg-muted"
                  )}
                  onClick={() => setSelectedFile(file.path)}
                >
                  <span className="text-sm">{getFileTypeIcon(file.type)}</span>
                  <span className="font-mono text-xs flex-1 truncate">{file.path}</span>
                  {file.diff && (
                    <div className="flex items-center space-x-1 text-xs">
                      <span className="text-green-600">+{file.diff.additions}</span>
                      <span className="text-red-600">-{file.diff.deletions}</span>
                    </div>
                  )}
                </motion.div>
              ))}
            </div>
          </ScrollArea>
        </div>

        {/* Diff viewer */}
        <div className="flex-1 overflow-hidden">
          {selectedFile ? (
            (() => {
              const selectedFileDiff = diff.modified_files.find(f => f.path === selectedFile);
              if (selectedFileDiff) {
                return (
                  <FileDiffView
                    fileDiff={selectedFileDiff}
                    settings={settings}
                    searchTerm={searchTerm}
                  />
                );
              } else {
                const fileType = diff.added_files.includes(selectedFile) ? 'added' : 'deleted';
                return (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center">
                      <FileText className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                      <h4 className="font-medium mb-2">File {fileType}</h4>
                      <p className="text-sm text-muted-foreground">
                        {fileType === 'added' ? 'This file was added in the checkpoint' : 'This file was deleted in the checkpoint'}
                      </p>
                    </div>
                  </div>
                );
              }
            })()
          ) : (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <GitCompare className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h4 className="font-medium mb-2">Select a file to view diff</h4>
                <p className="text-sm text-muted-foreground">
                  Choose a file from the sidebar to see the changes
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Settings Dialog */}
      <Dialog open={showSettings} onOpenChange={setShowSettings}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Diff Viewer Settings</DialogTitle>
          </DialogHeader>
          
          <div className="space-y-4 py-4">
            <div className="flex items-center justify-between">
              <Label htmlFor="line-numbers">Show Line Numbers</Label>
              <Switch
                id="line-numbers"
                checked={settings.showLineNumbers}
                onCheckedChange={(checked) => setSettings(prev => ({...prev, showLineNumbers: checked}))}
              />
            </div>
            
            <div className="flex items-center justify-between">
              <Label htmlFor="whitespace">Show Whitespace</Label>
              <Switch
                id="whitespace"
                checked={settings.showWhitespace}
                onCheckedChange={(checked) => setSettings(prev => ({...prev, showWhitespace: checked}))}
              />
            </div>
            
            <div className="flex items-center justify-between">
              <Label htmlFor="wrap-lines">Wrap Lines</Label>
              <Switch
                id="wrap-lines"
                checked={settings.wrapLines}
                onCheckedChange={(checked) => setSettings(prev => ({...prev, wrapLines: checked}))}
              />
            </div>
            
            <div className="flex items-center justify-between">
              <Label htmlFor="syntax">Syntax Highlighting</Label>
              <Switch
                id="syntax"
                checked={settings.highlightSyntax}
                onCheckedChange={(checked) => setSettings(prev => ({...prev, highlightSyntax: checked}))}
              />
            </div>
            
            <div className="space-y-2">
              <Label>Context Lines</Label>
              <Select value={settings.contextLines.toString()} onValueChange={(value) => setSettings(prev => ({...prev, contextLines: parseInt(value)}))}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="1">1 line</SelectItem>
                  <SelectItem value="3">3 lines</SelectItem>
                  <SelectItem value="5">5 lines</SelectItem>
                  <SelectItem value="10">10 lines</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};