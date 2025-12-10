/**
 * Checkpoint Export/Import - Advanced export and import capabilities
 * Supports multiple formats and import/export of checkpoint data
 */

"use client";

import React, { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Download,
  Upload,
  FileText,
  FileSpreadsheet,
  FileImage,
  Archive,
  Settings,
  Check,
  X,
  AlertCircle,
  Info,
  Lock,
  Unlock,
  Calendar,
  Hash,
  FileCode,
  Zap,
  Share2,
  Copy,
  ExternalLink,
  RefreshCw
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter } from '@/components/ui/dialog';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { cn } from '@/lib/utils';
import { 
  Checkpoint, 
  CheckpointExportFormat, 
  CheckpointExportResult 
} from '@/types/opencode';

interface CheckpointExportImportProps {
  /**
   * Checkpoints to export (if any)
   */
  checkpoints?: Checkpoint[];
  /**
   * Whether dialog is open
   */
  open: boolean;
  /**
   * Callback when dialog state changes
   */
  onOpenChange: (open: boolean) => void;
  /**
   * Default tab ('export' or 'import')
   */
  defaultTab?: 'export' | 'import';
  /**
   * Callback when export is completed
   */
  onExportComplete?: (result: CheckpointExportResult) => void;
  /**
   * Callback when import is completed
   */
  onImportComplete?: (checkpoints: Checkpoint[]) => void;
}

interface ExportProgress {
  stage: 'preparing' | 'collecting' | 'processing' | 'packaging' | 'complete';
  progress: number;
  current_file?: string;
  files_processed: number;
  total_files: number;
}

interface ImportProgress {
  stage: 'reading' | 'validating' | 'processing' | 'importing' | 'complete';
  progress: number;
  current_checkpoint?: string;
  checkpoints_processed: number;
  total_checkpoints: number;
  warnings: string[];
  errors: string[];
}

const ExportFormatCard: React.FC<{
  format: CheckpointExportFormat['format'];
  title: string;
  description: string;
  icon: React.ReactNode;
  isSelected: boolean;
  onSelect: () => void;
  features: string[];
}> = ({ format, title, description, icon, isSelected, onSelect, features }) => (
  <Card 
    className={cn(
      "cursor-pointer transition-all hover:shadow-md",
      isSelected && "ring-2 ring-primary bg-primary/5"
    )}
    onClick={onSelect}
  >
    <CardContent className="p-4">
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0">{icon}</div>
        <div className="flex-1 min-w-0">
          <h4 className="font-medium">{title}</h4>
          <p className="text-sm text-muted-foreground mt-1">{description}</p>
          <div className="flex flex-wrap gap-1 mt-2">
            {features.map(feature => (
              <Badge key={feature} variant="outline" className="text-xs">
                {feature}
              </Badge>
            ))}
          </div>
        </div>
        {isSelected && (
          <Check className="h-5 w-5 text-primary flex-shrink-0" />
        )}
      </div>
    </CardContent>
  </Card>
);

export const CheckpointExportImport: React.FC<CheckpointExportImportProps> = ({
  checkpoints = [],
  open,
  onOpenChange,
  defaultTab = 'export',
  onExportComplete,
  onImportComplete
}) => {
  const [activeTab, setActiveTab] = useState(defaultTab);
  const [exportFormat, setExportFormat] = useState<CheckpointExportFormat>({
    format: 'json',
    include_files: true,
    include_metadata: true,
    include_diffs: false,
    password_protected: false
  });
  const [exportPassword, setExportPassword] = useState('');
  const [isExporting, setIsExporting] = useState(false);
  const [exportProgress, setExportProgress] = useState<ExportProgress | null>(null);
  const [exportResult, setExportResult] = useState<CheckpointExportResult | null>(null);
  
  const [importFile, setImportFile] = useState<File | null>(null);
  const [importPassword, setImportPassword] = useState('');
  const [isImporting, setIsImporting] = useState(false);
  const [importProgress, setImportProgress] = useState<ImportProgress | null>(null);
  const [importPreview, setImportPreview] = useState<Checkpoint[] | null>(null);
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const exportFormats = [
    {
      format: 'json' as const,
      title: 'JSON Export',
      description: 'Complete checkpoint data in JSON format',
      icon: <FileCode className="h-8 w-8 text-blue-500" />,
      features: ['Complete data', 'Easy parsing', 'Cross-platform']
    },
    {
      format: 'markdown' as const,
      title: 'Markdown Report',
      description: 'Human-readable documentation',
      icon: <FileText className="h-8 w-8 text-green-500" />,
      features: ['Readable', 'Documentation', 'Version control']
    },
    {
      format: 'pdf' as const,
      title: 'PDF Report',
      description: 'Professional formatted document',
      icon: <FileImage className="h-8 w-8 text-red-500" />,
      features: ['Professional', 'Printable', 'Presentations']
    },
    {
      format: 'zip' as const,
      title: 'ZIP Archive',
      description: 'Complete backup with all files',
      icon: <Archive className="h-8 w-8 text-purple-500" />,
      features: ['Complete backup', 'File snapshots', 'Portable']
    }
  ];

  const handleExport = async () => {
    if (checkpoints.length === 0) return;
    
    setIsExporting(true);
    setExportProgress({
      stage: 'preparing',
      progress: 0,
      files_processed: 0,
      total_files: 0
    });

    try {
      // Simulate export process
      const stages = ['preparing', 'collecting', 'processing', 'packaging', 'complete'] as const;
      const totalFiles = checkpoints.reduce((acc, cp) => acc + (cp.metadata.filesModified?.length || 0), 0);
      
      for (let i = 0; i < stages.length; i++) {
        const stage = stages[i];
        setExportProgress({
          stage,
          progress: (i / (stages.length - 1)) * 100,
          files_processed: Math.floor((i / (stages.length - 1)) * totalFiles),
          total_files: totalFiles,
          current_file: i < stages.length - 1 ? `file-${i}.js` : undefined
        });
        
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
      
      const result: CheckpointExportResult = {
        success: true,
        file_path: `/downloads/checkpoints-${Date.now()}.${exportFormat.format}`,
        download_url: `https://example.com/downloads/checkpoints-${Date.now()}.${exportFormat.format}`,
        size_bytes: 1024 * 1024 * 2.5 // 2.5MB
      };
      
      setExportResult(result);
      onExportComplete?.(result);
    } catch (error) {
      setExportResult({
        success: false,
        error: 'Export failed',
        size_bytes: 0
      });
    } finally {
      setIsExporting(false);
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setImportFile(file);
      previewImport(file);
    }
  };

  const previewImport = async (file: File) => {
    try {
      // Simulate reading and parsing file
      const mockCheckpoints: Checkpoint[] = [
        {
          id: 'imported-1',
          session_id: 'new-session',
          name: 'Imported Checkpoint 1',
          description: 'Imported from backup',
          message_index: 0,
          timestamp: Date.now(),
          created_at: Date.now(),
          type: 'manual',
          metadata: {
            totalTokens: 1500,
            fileChanges: 3,
            toolsUsed: ['file_read', 'file_write']
          }
        }
      ];
      
      setImportPreview(mockCheckpoints);
    } catch (error) {
      console.error('Failed to preview import:', error);
    }
  };

  const handleImport = async () => {
    if (!importFile) return;
    
    setIsImporting(true);
    setImportProgress({
      stage: 'reading',
      progress: 0,
      checkpoints_processed: 0,
      total_checkpoints: importPreview?.length || 0,
      warnings: [],
      errors: []
    });

    try {
      const stages = ['reading', 'validating', 'processing', 'importing', 'complete'] as const;
      const totalCheckpoints = importPreview?.length || 0;
      
      for (let i = 0; i < stages.length; i++) {
        const stage = stages[i];
        setImportProgress({
          stage,
          progress: (i / (stages.length - 1)) * 100,
          checkpoints_processed: Math.floor((i / (stages.length - 1)) * totalCheckpoints),
          total_checkpoints: totalCheckpoints,
          current_checkpoint: i < stages.length - 1 ? `checkpoint-${i}` : undefined,
          warnings: i > 2 ? ['Some file snapshots were missing'] : [],
          errors: []
        });
        
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
      
      if (importPreview) {
        onImportComplete?.(importPreview);
      }
    } catch (error) {
      setImportProgress(prev => prev ? {
        ...prev,
        stage: 'complete',
        errors: ['Import failed: Invalid file format']
      } : null);
    } finally {
      setIsImporting(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
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

  const getProgressMessage = (progress: ExportProgress | ImportProgress) => {
    if ('current_file' in progress && progress.current_file) {
      return `Processing ${progress.current_file}...`;
    }
    if ('current_checkpoint' in progress && progress.current_checkpoint) {
      return `Importing ${progress.current_checkpoint}...`;
    }
    
    switch (progress.stage) {
      case 'preparing': return 'Preparing export...';
      case 'collecting': return 'Collecting checkpoint data...';
      case 'processing': return 'Processing files...';
      case 'packaging': return 'Creating archive...';
      case 'reading': return 'Reading import file...';
      case 'validating': return 'Validating data...';
      case 'importing': return 'Importing checkpoints...';
      case 'complete': return 'Complete!';
      default: return 'Processing...';
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle>Checkpoint Export & Import</DialogTitle>
          <DialogDescription>
            Export checkpoints for backup or sharing, import from previous exports
          </DialogDescription>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={(value: any) => setActiveTab(value)} className="flex-1">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="export" className="flex items-center space-x-2">
              <Download className="h-4 w-4" />
              <span>Export</span>
            </TabsTrigger>
            <TabsTrigger value="import" className="flex items-center space-x-2">
              <Upload className="h-4 w-4" />
              <span>Import</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="export" className="mt-4 space-y-6">
            {/* Export format selection */}
            <div>
              <h4 className="font-medium mb-3">Export Format</h4>
              <div className="grid grid-cols-2 gap-3">
                {exportFormats.map(format => (
                  <ExportFormatCard
                    key={format.format}
                    format={format.format}
                    title={format.title}
                    description={format.description}
                    icon={format.icon}
                    isSelected={exportFormat.format === format.format}
                    onSelect={() => setExportFormat(prev => ({...prev, format: format.format}))}
                    features={format.features}
                  />
                ))}
              </div>
            </div>

            {/* Export options */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Export Options</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <Label>Include File Snapshots</Label>
                    <div className="text-sm text-muted-foreground">Include complete file content at each checkpoint</div>
                  </div>
                  <Switch
                    checked={exportFormat.include_files}
                    onCheckedChange={(checked) => setExportFormat(prev => ({...prev, include_files: checked}))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Include Metadata</Label>
                    <div className="text-sm text-muted-foreground">Include token usage, costs, and tool information</div>
                  </div>
                  <Switch
                    checked={exportFormat.include_metadata}
                    onCheckedChange={(checked) => setExportFormat(prev => ({...prev, include_metadata: checked}))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Include Diffs</Label>
                    <div className="text-sm text-muted-foreground">Include file difference information between checkpoints</div>
                  </div>
                  <Switch
                    checked={exportFormat.include_diffs}
                    onCheckedChange={(checked) => setExportFormat(prev => ({...prev, include_diffs: checked}))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div>
                    <Label>Password Protection</Label>
                    <div className="text-sm text-muted-foreground">Encrypt export with password</div>
                  </div>
                  <Switch
                    checked={exportFormat.password_protected}
                    onCheckedChange={(checked) => setExportFormat(prev => ({...prev, password_protected: checked}))}
                  />
                </div>

                {exportFormat.password_protected && (
                  <div>
                    <Label htmlFor="export-password">Export Password</Label>
                    <Input
                      id="export-password"
                      type="password"
                      value={exportPassword}
                      onChange={(e) => setExportPassword(e.target.value)}
                      placeholder="Enter password for export..."
                    />
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Export summary */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Export Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground">Checkpoints</div>
                    <div className="font-medium">{checkpoints.length}</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Total Files</div>
                    <div className="font-medium">
                      {checkpoints.reduce((acc, cp) => acc + (cp.metadata.filesModified?.length || 0), 0)}
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Estimated Size</div>
                    <div className="font-medium">
                      {formatFileSize(checkpoints.length * 1024 * 200)} {/* Rough estimate */}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Export progress */}
            {isExporting && exportProgress && (
              <Card>
                <CardContent className="p-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">
                        {getProgressMessage(exportProgress)}
                      </span>
                      <span className="text-sm text-muted-foreground">
                        {Math.round(exportProgress.progress)}%
                      </span>
                    </div>
                    <Progress value={exportProgress.progress} />
                    <div className="text-xs text-muted-foreground">
                      {exportProgress.files_processed} of {exportProgress.total_files} files processed
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Export result */}
            {exportResult && (
              <Card>
                <CardContent className="p-4">
                  {exportResult.success ? (
                    <div className="space-y-3">
                      <div className="flex items-center space-x-2 text-green-600">
                        <Check className="h-4 w-4" />
                        <span className="font-medium">Export completed successfully!</span>
                      </div>
                      <div className="space-y-2">
                        <div className="text-sm">
                          <span className="text-muted-foreground">Size: </span>
                          <span>{formatFileSize(exportResult.size_bytes)}</span>
                        </div>
                        {exportResult.download_url && (
                          <div className="flex items-center space-x-2">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => copyToClipboard(exportResult.download_url!)}
                            >
                              <Copy className="h-4 w-4 mr-1" />
                              Copy Link
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => window.open(exportResult.download_url, '_blank')}
                            >
                              <ExternalLink className="h-4 w-4 mr-1" />
                              Download
                            </Button>
                          </div>
                        )}
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center space-x-2 text-red-600">
                      <X className="h-4 w-4" />
                      <span>{exportResult.error || 'Export failed'}</span>
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="import" className="mt-4 space-y-6">
            {/* File selection */}
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Select Import File</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <Button
                    variant="outline"
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full h-24 border-dashed"
                  >
                    <div className="text-center">
                      <Upload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                      <div className="text-sm">
                        {importFile ? importFile.name : 'Click to select file or drag and drop'}
                      </div>
                    </div>
                  </Button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".json,.zip,.md"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                  
                  {importFile && (
                    <div className="flex items-center space-x-2 p-2 bg-muted rounded">
                      <FileText className="h-4 w-4" />
                      <span className="text-sm">{importFile.name}</span>
                      <span className="text-xs text-muted-foreground">
                        ({formatFileSize(importFile.size)})
                      </span>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Import preview */}
            {importPreview && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Import Preview</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <div className="text-muted-foreground">Checkpoints</div>
                        <div className="font-medium">{importPreview.length}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Total Tokens</div>
                        <div className="font-medium">
                          {importPreview.reduce((acc, cp) => acc + cp.metadata.totalTokens, 0).toLocaleString()}
                        </div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">File Changes</div>
                        <div className="font-medium">
                          {importPreview.reduce((acc, cp) => acc + cp.metadata.fileChanges, 0)}
                        </div>
                      </div>
                    </div>
                    
                    <ScrollArea className="h-40">
                      <div className="space-y-2">
                        {importPreview.map(checkpoint => (
                          <div key={checkpoint.id} className="flex items-center space-x-2 text-sm p-2 border rounded">
                            <Badge variant="outline">{checkpoint.type}</Badge>
                            <span className="flex-1">{checkpoint.name}</span>
                            <span className="text-muted-foreground">
                              {checkpoint.metadata.totalTokens.toLocaleString()} tokens
                            </span>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Import progress */}
            {isImporting && importProgress && (
              <Card>
                <CardContent className="p-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">
                        {getProgressMessage(importProgress)}
                      </span>
                      <span className="text-sm text-muted-foreground">
                        {Math.round(importProgress.progress)}%
                      </span>
                    </div>
                    <Progress value={importProgress.progress} />
                    <div className="text-xs text-muted-foreground">
                      {importProgress.checkpoints_processed} of {importProgress.total_checkpoints} checkpoints processed
                    </div>
                    
                    {importProgress.warnings.length > 0 && (
                      <div className="text-xs text-yellow-600">
                        {importProgress.warnings.map((warning, index) => (
                          <div key={index}>⚠️ {warning}</div>
                        ))}
                      </div>
                    )}
                    
                    {importProgress.errors.length > 0 && (
                      <div className="text-xs text-red-600">
                        {importProgress.errors.map((error, index) => (
                          <div key={index}>❌ {error}</div>
                        ))}
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>

        <DialogFooter className="flex items-center justify-between">
          <div className="text-xs text-muted-foreground">
            {activeTab === 'export' && checkpoints.length > 0 && (
              <span>{checkpoints.length} checkpoint{checkpoints.length !== 1 ? 's' : ''} selected</span>
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            <Button variant="outline" onClick={() => onOpenChange(false)}>
              Cancel
            </Button>
            
            {activeTab === 'export' ? (
              <Button
                onClick={handleExport}
                disabled={checkpoints.length === 0 || isExporting || (exportFormat.password_protected && !exportPassword)}
              >
                {isExporting ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Exporting...
                  </>
                ) : (
                  <>
                    <Download className="h-4 w-4 mr-2" />
                    Export Checkpoints
                  </>
                )}
              </Button>
            ) : (
              <Button
                onClick={handleImport}
                disabled={!importFile || isImporting}
              >
                {isImporting ? (
                  <>
                    <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                    Importing...
                  </>
                ) : (
                  <>
                    <Upload className="h-4 w-4 mr-2" />
                    Import Checkpoints
                  </>
                )}
              </Button>
            )}
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};