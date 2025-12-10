import React, { useState } from "react";
import { 
  CheckCircle2, 
  Circle, 
  Clock,
  FolderOpen,
  FileText,
  Search,
  Terminal,
  FileEdit,
  Code,
  ChevronRight,
  Maximize2,
  GitBranch,
  X,
  Info,
  AlertCircle,
  Settings,
  Fingerprint,
  Cpu,
  FolderSearch,
  List,
  LogOut,
  Edit3,
  FilePlus,
  Book,
  BookOpen,
  Globe,
  ListChecks,
  ListPlus,
  Globe2,
  Package,
  ChevronDown,
  Package2,
  Wrench,
  CheckSquare,
  type LucideIcon,
  Sparkles,
  Bot,
  Zap,
  FileCode,
  Folder,
  ChevronUp,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Button } from "@/components/ui/button";
import { createPortal } from "react-dom";
import * as Diff from 'diff';
import { Card, CardContent } from "@/components/ui/card";
import ReactMarkdown from "react-markdown";

/**
 * Widget for TodoWrite tool - displays a beautiful TODO list
 */
export const TodoWidget: React.FC<{ todos: any[]; result?: any }> = ({ todos, result: _result }) => {
  const statusIcons = {
    completed: <CheckCircle2 className="h-4 w-4 text-green-500" />,
    in_progress: <Clock className="h-4 w-4 text-blue-500 animate-pulse" />,
    pending: <Circle className="h-4 w-4 text-muted-foreground" />
  };

  const priorityColors = {
    high: "bg-red-500/10 text-red-500 border-red-500/20",
    medium: "bg-yellow-500/10 text-yellow-500 border-yellow-500/20",
    low: "bg-green-500/10 text-green-500 border-green-500/20"
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 mb-3">
        <FileEdit className="h-4 w-4 text-primary" />
        <span className="text-sm font-medium">Todo List</span>
      </div>
      <div className="space-y-2">
        {todos.map((todo, idx) => (
          <div
            key={todo.id || idx}
            className={cn(
              "flex items-start gap-3 p-3 rounded-lg border bg-card/50",
              todo.status === "completed" && "opacity-60"
            )}
          >
            <div className="mt-0.5">
              {statusIcons[todo.status as keyof typeof statusIcons] || statusIcons.pending}
            </div>
            <div className="flex-1 space-y-1">
              <p className={cn(
                "text-sm",
                todo.status === "completed" && "line-through"
              )}>
                {todo.content}
              </p>
              {todo.priority && (
                <Badge 
                  variant="outline" 
                  className={cn("text-xs", priorityColors[todo.priority as keyof typeof priorityColors])}
                >
                  {todo.priority}
                </Badge>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * Widget for LS (List Directory) tool
 */
export const LSWidget: React.FC<{ path: string; result?: any }> = ({ path, result }) => {
  // If we have a result, show it using the LSResultWidget
  if (result) {
    let resultContent = '';
    if (typeof result.content === 'string') {
      resultContent = result.content;
    } else if (result.content && typeof result.content === 'object') {
      if (result.content.text) {
        resultContent = result.content.text;
      } else if (Array.isArray(result.content)) {
        resultContent = result.content
          .map((c: any) => (typeof c === 'string' ? c : c.text || JSON.stringify(c)))
          .join('\n');
      } else {
        resultContent = JSON.stringify(result.content, null, 2);
      }
    }
    
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2 p-3 rounded-lg bg-muted/50">
          <FolderOpen className="h-4 w-4 text-primary" />
          <span className="text-sm">Directory contents for:</span>
          <code className="text-sm font-mono bg-background px-2 py-0.5 rounded">
            {path}
          </code>
        </div>
        {resultContent && <LSResultWidget content={resultContent} />}
      </div>
    );
  }
  
  return (
    <div className="flex items-center gap-2 p-3 rounded-lg bg-muted/50">
      <FolderOpen className="h-4 w-4 text-primary" />
      <span className="text-sm">Listing directory:</span>
      <code className="text-sm font-mono bg-background px-2 py-0.5 rounded">
        {path}
      </code>
      {!result && (
        <div className="ml-auto flex items-center gap-1 text-xs text-muted-foreground">
          <div className="h-2 w-2 bg-blue-500 rounded-full animate-pulse" />
          <span>Loading...</span>
        </div>
      )}
    </div>
  );
};

/**
 * Widget for LS tool result - displays directory tree structure
 */
export const LSResultWidget: React.FC<{ content: string }> = ({ content }) => {
  const [expandedDirs, setExpandedDirs] = useState<Set<string>>(new Set());
  
  // Parse the directory tree structure
  const parseDirectoryTree = (rawContent: string) => {
    const lines = rawContent.split('\n');
    const entries: Array<{
      path: string;
      name: string;
      type: 'file' | 'directory';
      level: number;
    }> = [];
    
    let currentPath: string[] = [];
    
    for (const line of lines) {
      // Skip NOTE section and everything after it
      if (line.startsWith('NOTE:')) {
        break;
      }
      
      // Skip empty lines
      if (!line.trim()) continue;
      
      // Calculate indentation level
      const indent = line.match(/^(\s*)/)?.[1] || '';
      const level = Math.floor(indent.length / 2);
      
      // Extract the entry name
      const entryMatch = line.match(/^\s*-\s+(.+?)(\/$)?$/);
      if (!entryMatch) continue;
      
      const fullName = entryMatch[1];
      const isDirectory = line.trim().endsWith('/');
      const name = isDirectory ? fullName : fullName;
      
      // Update current path based on level
      currentPath = currentPath.slice(0, level);
      currentPath.push(name);
      
      entries.push({
        path: currentPath.join('/'),
        name,
        type: isDirectory ? 'directory' : 'file',
        level,
      });
    }
    
    return entries;
  };
  
  const entries = parseDirectoryTree(content);
  
  const toggleDirectory = (path: string) => {
    setExpandedDirs(prev => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };
  
  // Group entries by parent for collapsible display
  const getChildren = (parentPath: string, parentLevel: number) => {
    return entries.filter(e => {
      if (e.level !== parentLevel + 1) return false;
      const parentParts = parentPath.split('/').filter(Boolean);
      const entryParts = e.path.split('/').filter(Boolean);
      
      // Check if this entry is a direct child of the parent
      if (entryParts.length !== parentParts.length + 1) return false;
      
      // Check if all parent parts match
      for (let i = 0; i < parentParts.length; i++) {
        if (parentParts[i] !== entryParts[i]) return false;
      }
      
      return true;
    });
  };
  
  const renderEntry = (entry: typeof entries[0], isRoot = false) => {
    const hasChildren = entry.type === 'directory' && 
      entries.some(e => e.path.startsWith(entry.path + '/') && e.level === entry.level + 1);
    const isExpanded = expandedDirs.has(entry.path) || isRoot;
    
    const getIcon = () => {
      if (entry.type === 'directory') {
        return isExpanded ? 
          <FolderOpen className="h-3.5 w-3.5 text-blue-500" /> : 
          <Folder className="h-3.5 w-3.5 text-blue-500" />;
      }
      
      // File type icons based on extension
      const ext = entry.name.split('.').pop()?.toLowerCase();
      switch (ext) {
        case 'rs':
          return <FileCode className="h-3.5 w-3.5 text-orange-500" />;
        case 'toml':
        case 'yaml':
        case 'yml':
        case 'json':
          return <FileText className="h-3.5 w-3.5 text-yellow-500" />;
        case 'md':
          return <FileText className="h-3.5 w-3.5 text-blue-400" />;
        case 'js':
        case 'jsx':
        case 'ts':
        case 'tsx':
          return <FileCode className="h-3.5 w-3.5 text-yellow-400" />;
        case 'py':
          return <FileCode className="h-3.5 w-3.5 text-blue-500" />;
        case 'go':
          return <FileCode className="h-3.5 w-3.5 text-cyan-500" />;
        case 'sh':
        case 'bash':
          return <Terminal className="h-3.5 w-3.5 text-green-500" />;
        default:
          return <FileText className="h-3.5 w-3.5 text-muted-foreground" />;
      }
    };
    
    return (
      <div key={entry.path}>
        <div 
          className={cn(
            "flex items-center gap-2 py-1 px-2 rounded hover:bg-muted/50 transition-colors cursor-pointer",
            !isRoot && "ml-4"
          )}
          onClick={() => entry.type === 'directory' && hasChildren && toggleDirectory(entry.path)}
        >
          {entry.type === 'directory' && hasChildren && (
            <ChevronRight className={cn(
              "h-3 w-3 text-muted-foreground transition-transform",
              isExpanded && "rotate-90"
            )} />
          )}
          {(!hasChildren || entry.type !== 'directory') && (
            <div className="w-3" />
          )}
          {getIcon()}
          <span className="text-sm font-mono">{entry.name}</span>
        </div>
        
        {entry.type === 'directory' && hasChildren && isExpanded && (
          <div className="ml-2">
            {getChildren(entry.path, entry.level).map(child => renderEntry(child))}
          </div>
        )}
      </div>
    );
  };
  
  // Get root entries
  const rootEntries = entries.filter(e => e.level === 0);
  
  return (
    <div className="rounded-lg border bg-muted/20 p-3">
      <div className="space-y-1">
        {rootEntries.map(entry => renderEntry(entry, true))}
      </div>
    </div>
  );
};

/**
 * Widget for Read tool
 */
export const ReadWidget: React.FC<{ filePath: string; result?: any }> = ({ filePath, result }) => {
  // If we have a result, show it using the ReadResultWidget
  if (result) {
    let resultContent = '';
    if (typeof result.content === 'string') {
      resultContent = result.content;
    } else if (result.content && typeof result.content === 'object') {
      if (result.content.text) {
        resultContent = result.content.text;
      } else if (Array.isArray(result.content)) {
        resultContent = result.content
          .map((c: any) => (typeof c === 'string' ? c : c.text || JSON.stringify(c)))
          .join('\n');
      } else {
        resultContent = JSON.stringify(result.content, null, 2);
      }
    }
    
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2 p-3 rounded-lg bg-muted/50">
          <FileText className="h-4 w-4 text-primary" />
          <span className="text-sm">File content:</span>
          <code className="text-sm font-mono bg-background px-2 py-0.5 rounded flex-1 truncate">
            {filePath}
          </code>
        </div>
        {resultContent && <ReadResultWidget content={resultContent} filePath={filePath} />}
      </div>
    );
  }
  
  return (
    <div className="flex items-center gap-2 p-3 rounded-lg bg-muted/50">
      <FileText className="h-4 w-4 text-primary" />
      <span className="text-sm">Reading file:</span>
      <code className="text-sm font-mono bg-background px-2 py-0.5 rounded flex-1 truncate">
        {filePath}
      </code>
      {!result && (
        <div className="ml-auto flex items-center gap-1 text-xs text-muted-foreground">
          <div className="h-2 w-2 bg-blue-500 rounded-full animate-pulse" />
          <span>Loading...</span>
        </div>
      )}
    </div>
  );
};

/**
 * Widget for Read tool result - shows file content with line numbers
 */
export const ReadResultWidget: React.FC<{ content: string; filePath?: string }> = ({ content, filePath }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Extract file extension for syntax highlighting
  const getLanguage = (path?: string) => {
    if (!path) return "text";
    const ext = path.split('.').pop()?.toLowerCase();
    const languageMap: Record<string, string> = {
      ts: "typescript",
      tsx: "tsx",
      js: "javascript",
      jsx: "jsx",
      py: "python",
      rs: "rust",
      go: "go",
      java: "java",
      cpp: "cpp",
      c: "c",
      cs: "csharp",
      php: "php",
      rb: "ruby",
      swift: "swift",
      kt: "kotlin",
      scala: "scala",
      sh: "bash",
      bash: "bash",
      zsh: "bash",
      yaml: "yaml",
      yml: "yaml",
      json: "json",
      xml: "xml",
      html: "html",
      css: "css",
      scss: "scss",
      sass: "sass",
      less: "less",
      sql: "sql",
      md: "markdown",
      toml: "ini",
      ini: "ini",
      dockerfile: "dockerfile",
      makefile: "makefile"
    };
    return languageMap[ext || ""] || "text";
  };

  // Parse content to separate line numbers from code
  const parseContent = (rawContent: string) => {
    const lines = rawContent.split('\n');
    const codeLines: string[] = [];
    let minLineNumber = Infinity;

    // First, determine if the content is likely a numbered list from the 'read' tool.
    const nonEmptyLines = lines.filter(line => line.trim() !== '');
    if (nonEmptyLines.length === 0) {
      return { codeContent: rawContent, startLineNumber: 1 };
    }
    const parsableLines = nonEmptyLines.filter(line => /^\s*\d+→/.test(line)).length;
    const isLikelyNumbered = (parsableLines / nonEmptyLines.length) > 0.5;

    if (!isLikelyNumbered) {
      return { codeContent: rawContent, startLineNumber: 1 };
    }
    
    // If it's a numbered list, parse it strictly.
    for (const line of lines) {
      // Remove leading whitespace before parsing
      const trimmedLine = line.trimStart();
      const match = trimmedLine.match(/^(\d+)→(.*)$/);
      if (match) {
        const lineNum = parseInt(match[1], 10);
        if (minLineNumber === Infinity) {
          minLineNumber = lineNum;
        }
        // Preserve the code content exactly as it appears after the arrow
        codeLines.push(match[2]);
      } else if (line.trim() === '') {
        // Preserve empty lines
        codeLines.push('');
      } else {
        // If a line in a numbered block does not match, it's a formatting anomaly.
        codeLines.push('');
      }
    }
    
    // Remove trailing empty lines
    while (codeLines.length > 0 && codeLines[codeLines.length - 1] === '') {
      codeLines.pop();
    }
    
    return {
      codeContent: codeLines.join('\n'),
      startLineNumber: minLineNumber === Infinity ? 1 : minLineNumber
    };
  };

  const language = getLanguage(filePath);
  const { codeContent, startLineNumber } = parseContent(content);
  const lineCount = content.split('\n').filter(line => line.trim()).length;
  const isLargeFile = lineCount > 20;

  return (
    <div className="rounded-lg overflow-hidden border bg-zinc-950 w-full">
      <div className="px-4 py-2 border-b bg-zinc-900/50 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileText className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-xs font-mono text-muted-foreground">
            {filePath || "File content"}
          </span>
          {isLargeFile && (
            <span className="text-xs text-muted-foreground">
              ({lineCount} lines)
            </span>
          )}
        </div>
        {isLargeFile && (
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            <ChevronRight className={cn("h-3 w-3 transition-transform", isExpanded && "rotate-90")} />
            {isExpanded ? "Collapse" : "Expand"}
          </button>
        )}
      </div>
      
      {(!isLargeFile || isExpanded) && (
        <div className="relative overflow-x-auto">
          <SyntaxHighlighter
            language={language}
            style={oneDark}
            showLineNumbers
            startingLineNumber={startLineNumber}
            wrapLongLines={false}
            customStyle={{
              margin: 0,
              background: 'transparent',
              lineHeight: '1.6'
            }}
            codeTagProps={{
              style: {
                fontSize: '0.75rem'
              }
            }}
            lineNumberStyle={{
              minWidth: "3.5rem",
              paddingRight: "1rem",
              textAlign: "right",
              opacity: 0.5,
            }}
          >
            {codeContent}
          </SyntaxHighlighter>
        </div>
      )}
      
      {isLargeFile && !isExpanded && (
        <div className="px-4 py-3 text-xs text-muted-foreground text-center bg-zinc-900/30">
          Click &quot;Expand&quot; to view the full file
        </div>
      )}
    </div>
  );
};

// Export all the remaining widgets from the original file
export {
  GlobWidget,
  BashWidget,
  WriteWidget,
  GrepWidget,
  EditWidget,
  EditResultWidget,
  MCPWidget,
  CommandWidget,
  CommandOutputWidget,
  SummaryWidget,
  MultiEditWidget,
  MultiEditResultWidget,
  SystemReminderWidget,
  SystemInitializedWidget,
  TaskWidget,
  ThinkingWidget,
  WebSearchWidget
} from "./tool-widgets-extended";