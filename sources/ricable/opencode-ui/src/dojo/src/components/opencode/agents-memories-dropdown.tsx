import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronDown, Edit2, Brain, Loader2, Globe, User, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface AgentsFile {
  path: string;
  relative_path: string;
  absolute_path: string;
  scope: "global" | "project";
  exists: boolean;
  size: number;
  modified: number;
}

interface AgentsMemoriesDropdownProps {
  /**
   * The project path to search for AGENTS.md files
   */
  projectPath: string;
  /**
   * Callback when an edit button is clicked
   */
  onEditFile: (file: AgentsFile) => void;
  /**
   * Callback when create new file is clicked
   */
  onCreateFile?: (scope: "global" | "project") => void;
  /**
   * Optional className for styling
   */
  className?: string;
}

// Mock API functions - replace with actual OpenCode API calls
const mockApi = {
  async findAgentsFiles(projectPath: string): Promise<AgentsFile[]> {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 300));
    
    // Return mock AGENTS.md files
    return [
      {
        path: "~/.config/opencode/AGENTS.md",
        relative_path: "~/.config/opencode/AGENTS.md",
        absolute_path: "/Users/example/.config/opencode/AGENTS.md",
        scope: "global",
        exists: true,
        size: 2048,
        modified: Date.now() - 86400000 // 1 day ago
      },
      {
        path: "AGENTS.md",
        relative_path: "AGENTS.md",
        absolute_path: projectPath + "/AGENTS.md",
        scope: "project",
        exists: true,
        size: 1024,
        modified: Date.now() - 3600000 // 1 hour ago
      }
    ];
  }
};

/**
 * Format Unix timestamp to relative time
 */
const formatUnixTimestamp = (timestamp: number): string => {
  const now = Date.now();
  const diff = now - timestamp;
  
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  
  if (days > 0) return `${days} day${days > 1 ? 's' : ''} ago`;
  if (hours > 0) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
  if (minutes > 0) return `${minutes} min${minutes > 1 ? 's' : ''} ago`;
  return 'just now';
};

/**
 * AgentsMemoriesDropdown component - Shows all AGENTS.md files for OpenCode
 * Adapted from ClaudeMemoriesDropdown for OpenCode's multi-scope agent system
 * 
 * @example
 * <AgentsMemoriesDropdown
 *   projectPath="/Users/example/project"
 *   onEditFile={(file) => console.log('Edit file:', file)}
 *   onCreateFile={(scope) => console.log('Create new file:', scope)}
 * />
 */
export const AgentsMemoriesDropdown: React.FC<AgentsMemoriesDropdownProps> = ({
  projectPath,
  onEditFile,
  onCreateFile,
  className,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [files, setFiles] = useState<AgentsFile[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Load AGENTS.md files when dropdown opens
  useEffect(() => {
    if (isOpen && files.length === 0) {
      loadAgentsFiles();
    }
  }, [isOpen]);
  
  const loadAgentsFiles = async () => {
    try {
      setLoading(true);
      setError(null);
      const foundFiles = await mockApi.findAgentsFiles(projectPath);
      setFiles(foundFiles);
    } catch (err) {
      console.error("Failed to load AGENTS.md files:", err);
      setError("Failed to load AGENTS.md files");
    } finally {
      setLoading(false);
    }
  };
  
  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const getScopeIcon = (scope: "global" | "project") => {
    return scope === "global" ? (
      <Globe className="h-3 w-3" />
    ) : (
      <User className="h-3 w-3" />
    );
  };

  const getScopeColor = (scope: "global" | "project") => {
    return scope === "global" ? "default" : "secondary";
  };
  
  return (
    <div className={cn("w-full", className)}>
      <Card className="overflow-hidden">
        {/* Dropdown Header */}
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="w-full flex items-center justify-between p-3 hover:bg-accent/50 transition-colors"
        >
          <div className="flex items-center space-x-2">
            <Brain className="h-4 w-4 text-blue-500" />
            <span className="text-sm font-medium">Agent Memories</span>
            {files.length > 0 && !loading && (
              <span className="text-xs text-muted-foreground">({files.length})</span>
            )}
          </div>
          <motion.div
            animate={{ rotate: isOpen ? 180 : 0 }}
            transition={{ duration: 0.2 }}
          >
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          </motion.div>
        </button>
        
        {/* Dropdown Content */}
        <AnimatePresence>
          {isOpen && (
            <motion.div
              initial={{ height: 0 }}
              animate={{ height: "auto" }}
              exit={{ height: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden"
            >
              <div className="border-t border-border">
                {loading ? (
                  <div className="p-4 flex items-center justify-center">
                    <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                  </div>
                ) : error ? (
                  <div className="p-3 text-xs text-destructive">{error}</div>
                ) : (
                  <>
                    {/* File List */}
                    {files.length === 0 ? (
                      <div className="p-3 text-xs text-muted-foreground text-center">
                        No AGENTS.md files found
                      </div>
                    ) : (
                      <div className="max-h-64 overflow-y-auto">
                        {files.map((file, index) => (
                          <motion.div
                            key={file.absolute_path}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.05 }}
                            className="flex items-center justify-between p-3 hover:bg-accent/50 transition-colors border-b border-border last:border-b-0"
                          >
                            <div className="flex-1 min-w-0 mr-2">
                              <div className="flex items-center gap-2 mb-1">
                                <p className="text-xs font-mono truncate">{file.relative_path}</p>
                                <Badge 
                                  variant={getScopeColor(file.scope)} 
                                  className="text-xs px-1.5 py-0.5 h-4 flex items-center gap-1"
                                >
                                  {getScopeIcon(file.scope)}
                                  {file.scope}
                                </Badge>
                              </div>
                              <div className="flex items-center space-x-3">
                                <span className="text-xs text-muted-foreground">
                                  {formatFileSize(file.size)}
                                </span>
                                <span className="text-xs text-muted-foreground">
                                  Modified {formatUnixTimestamp(file.modified)}
                                </span>
                              </div>
                            </div>
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-7 w-7 flex-shrink-0"
                              onClick={(e) => {
                                e.stopPropagation();
                                onEditFile(file);
                              }}
                            >
                              <Edit2 className="h-3 w-3" />
                            </Button>
                          </motion.div>
                        ))}
                      </div>
                    )}

                    {/* Create New File Options */}
                    {onCreateFile && (
                      <div className="border-t border-border p-2 bg-muted/30">
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-muted-foreground">Create new:</span>
                          <div className="flex gap-1">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => onCreateFile("global")}
                              className="h-6 px-2 text-xs"
                            >
                              <Plus className="h-3 w-3 mr-1" />
                              <Globe className="h-3 w-3 mr-1" />
                              Global
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => onCreateFile("project")}
                              className="h-6 px-2 text-xs"
                            >
                              <Plus className="h-3 w-3 mr-1" />
                              <User className="h-3 w-3 mr-1" />
                              Project
                            </Button>
                          </div>
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>
    </div>
  );
};