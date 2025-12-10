import React, { useState } from "react";
import { 
  Search,
  Terminal,
  FileEdit,
  Code,
  ChevronRight,
  GitBranch,
  Info,
  AlertCircle,
  Package,
  ChevronDown,
  Package2,
  Sparkles,
  Bot,
  Zap,
  ChevronUp,
  Globe,
  Globe2
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import * as Diff from 'diff';
import { Card, CardContent } from "@/components/ui/card";
import ReactMarkdown from "react-markdown";

/**
 * Widget for Glob tool
 */
export const GlobWidget: React.FC<{ pattern: string; result?: any }> = ({ pattern, result }) => {
  // Extract result content if available
  let resultContent = '';
  let isError = false;
  
  if (result) {
    isError = result.is_error || false;
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
  }
  
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 p-3 rounded-lg bg-muted/50">
        <Search className="h-4 w-4 text-primary" />
        <span className="text-sm">Searching for pattern:</span>
        <code className="text-sm font-mono bg-background px-2 py-0.5 rounded">
          {pattern}
        </code>
        {!result && (
          <div className="ml-auto flex items-center gap-1 text-xs text-muted-foreground">
            <div className="h-2 w-2 bg-blue-500 rounded-full animate-pulse" />
            <span>Searching...</span>
          </div>
        )}
      </div>
      
      {/* Show result if available */}
      {result && (
        <div className={cn(
          "p-3 rounded-md border text-xs font-mono whitespace-pre-wrap overflow-x-auto",
          isError 
            ? "border-red-500/20 bg-red-500/5 text-red-400" 
            : "border-green-500/20 bg-green-500/5 text-green-300"
        )}>
          {resultContent || (isError ? "Search failed" : "No matches found")}
        </div>
      )}
    </div>
  );
};

/**
 * Widget for Bash tool
 */
export const BashWidget: React.FC<{ 
  command: string; 
  description?: string;
  result?: any;
}> = ({ command, description, result }) => {
  // Extract result content if available
  let resultContent = '';
  let isError = false;
  
  if (result) {
    isError = result.is_error || false;
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
  }
  
  return (
    <div className="rounded-lg border bg-zinc-950 overflow-hidden">
      <div className="px-4 py-2 bg-zinc-900/50 flex items-center gap-2 border-b">
        <Terminal className="h-3.5 w-3.5 text-green-500" />
        <span className="text-xs font-mono text-muted-foreground">Terminal</span>
        {description && (
          <>
            <ChevronRight className="h-3 w-3 text-muted-foreground" />
            <span className="text-xs text-muted-foreground">{description}</span>
          </>
        )}
        {/* Show loading indicator when no result yet */}
        {!result && (
          <div className="ml-auto flex items-center gap-1 text-xs text-muted-foreground">
            <div className="h-2 w-2 bg-green-500 rounded-full animate-pulse" />
            <span>Running...</span>
          </div>
        )}
      </div>
      <div className="p-4 space-y-3">
        <code className="text-xs font-mono text-green-400 block">
          $ {command}
        </code>
        
        {/* Show result if available */}
        {result && (
          <div className={cn(
            "mt-3 p-3 rounded-md border text-xs font-mono whitespace-pre-wrap overflow-x-auto",
            isError 
              ? "border-red-500/20 bg-red-500/5 text-red-400" 
              : "border-green-500/20 bg-green-500/5 text-green-300"
          )}>
            {resultContent || (isError ? "Command failed" : "Command completed")}
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Widget for Write tool
 */
export const WriteWidget: React.FC<{ filePath: string; content: string; result?: any }> = ({ filePath, content, result: _result }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Extract file extension for syntax highlighting
  const getLanguage = (path: string) => {
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

  const language = getLanguage(filePath);
  const isLargeContent = content.length > 1000;
  const displayContent = isLargeContent ? content.substring(0, 1000) + "\n..." : content;

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 p-3 rounded-lg bg-muted/50">
        <FileEdit className="h-4 w-4 text-primary" />
        <span className="text-sm">Writing to file:</span>
        <code className="text-sm font-mono bg-background px-2 py-0.5 rounded flex-1 truncate">
          {filePath}
        </code>
      </div>
      <div className="rounded-lg border bg-zinc-950 overflow-hidden">
        <div className="px-4 py-2 border-b bg-zinc-950 flex items-center justify-between">
          <span className="text-xs font-mono text-muted-foreground">Preview</span>
          {isLargeContent && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              {isExpanded ? "Collapse" : "Expand"}
            </button>
          )}
        </div>
        <div className="overflow-auto max-h-[400px]">
          <SyntaxHighlighter
            language={language}
            style={oneDark}
            customStyle={{
              margin: 0,
              padding: '1rem',
              background: 'transparent',
              fontSize: '0.75rem',
              lineHeight: '1.5'
            }}
          >
            {isExpanded ? content : displayContent}
          </SyntaxHighlighter>
        </div>
      </div>
    </div>
  );
};

/**
 * Widget for Grep tool
 */
export const GrepWidget: React.FC<{ 
  pattern: string; 
  include?: string; 
  path?: string;
  exclude?: string;
  result?: any;
}> = ({ pattern, include, path, exclude, result }) => {
  // Extract result content if available
  let resultContent = '';
  let isError = false;
  
  if (result) {
    isError = result.is_error || false;
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
  }
  
  // Parse grep results to extract file paths and matches
  const parseGrepResults = (content: string) => {
    const lines = content.split('\n').filter(line => line.trim());
    const results: Array<{
      file: string;
      lineNumber: number;
      content: string;
    }> = [];
    
    lines.forEach(line => {
      // Common grep output format: filename:lineNumber:content
      const match = line.match(/^(.+?):(\d+):(.*)$/);
      if (match) {
        results.push({
          file: match[1],
          lineNumber: parseInt(match[2], 10),
          content: match[3]
        });
      }
    });
    
    return results;
  };
  
  const grepResults = result && !isError ? parseGrepResults(resultContent) : [];
  
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 p-3 rounded-lg bg-gradient-to-r from-emerald-500/10 to-teal-500/10 border border-emerald-500/20">
        <Search className="h-4 w-4 text-emerald-500" />
        <span className="text-sm font-medium">Searching with grep</span>
        {!result && (
          <div className="ml-auto flex items-center gap-1 text-xs text-muted-foreground">
            <div className="h-2 w-2 bg-emerald-500 rounded-full animate-pulse" />
            <span>Searching...</span>
          </div>
        )}
      </div>
      
      {/* Search Parameters */}
      <div className="rounded-lg border bg-muted/20 p-3 space-y-2">
        <div className="grid gap-2">
          {/* Pattern */}
          <div className="flex items-center gap-3">
            <Code className="h-3 w-3 text-emerald-500" />
            <span className="text-xs font-medium text-muted-foreground">Pattern:</span>
            <code className="font-mono text-sm bg-emerald-500/10 border border-emerald-500/20 px-3 py-1.5 rounded-md text-emerald-600 dark:text-emerald-400">
              {pattern}
            </code>
          </div>
          
          {/* Additional parameters */}
          {(include || exclude || path) && (
            <div className="text-xs text-muted-foreground space-y-1">
              {path && <div>Path: {path}</div>}
              {include && <div>Include: {include}</div>}
              {exclude && <div>Exclude: {exclude}</div>}
            </div>
          )}
        </div>
      </div>
      
      {/* Results */}
      {result && (
        <div className="space-y-2">
          {isError ? (
            <div className="flex items-center gap-3 p-4 rounded-lg bg-red-500/10 border border-red-500/20">
              <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
              <div className="text-sm text-red-600 dark:text-red-400">
                {resultContent || "Search failed"}
              </div>
            </div>
          ) : grepResults.length > 0 ? (
            <div className="rounded-lg border bg-zinc-950 overflow-hidden">
              <div className="max-h-[300px] overflow-y-auto">
                {grepResults.map((match, idx) => (
                  <div key={idx} className="flex items-start gap-3 p-3 border-b border-zinc-800 hover:bg-zinc-900/50">
                    <span className="text-xs font-mono text-emerald-400">{match.lineNumber}</span>
                    <div className="flex-1 space-y-1">
                      <div className="text-xs font-medium text-blue-400">{match.file}</div>
                      <code className="text-xs font-mono text-zinc-300 block">{match.content}</code>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-3 p-4 rounded-lg bg-amber-500/10 border border-amber-500/20">
              <Info className="h-5 w-5 text-amber-500 flex-shrink-0" />
              <div className="text-sm text-amber-600 dark:text-amber-400">
                No matches found for the given pattern.
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

/**
 * Widget for Edit tool - shows the edit operation
 */
export const EditWidget: React.FC<{ 
  file_path: string; 
  old_string: string; 
  new_string: string;
  result?: any;
}> = ({ file_path, old_string, new_string, result: _result }) => {
  const diffResult = Diff.diffLines(old_string || '', new_string || '', { 
    newlineIsToken: true,
    ignoreWhitespace: false 
  });

  const getLanguage = (path: string) => {
    const ext = path.split('.').pop()?.toLowerCase();
    const languageMap: Record<string, string> = {
      ts: "typescript", tsx: "tsx", js: "javascript", jsx: "jsx",
      py: "python", rs: "rust", go: "go", java: "java"
    };
    return languageMap[ext || ""] || "text";
  };

  const language = getLanguage(file_path);

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 mb-2">
        <FileEdit className="h-4 w-4 text-primary" />
        <span className="text-sm font-medium">Applying Edit to:</span>
        <code className="text-sm font-mono bg-background px-2 py-0.5 rounded flex-1 truncate">
          {file_path}
        </code>
      </div>

      <div className="rounded-lg border bg-zinc-950 overflow-hidden text-xs font-mono">
        <div className="max-h-[440px] overflow-y-auto overflow-x-auto">
          {diffResult.map((part, index) => {
            const partClass = part.added 
              ? 'bg-green-950/20' 
              : part.removed 
              ? 'bg-red-950/20'
              : '';
            
            const value = part.value.endsWith('\n') ? part.value.slice(0, -1) : part.value;

            return (
              <div key={index} className={cn(partClass, "flex")}>
                <div className="w-8 select-none text-center flex-shrink-0">
                  {part.added ? <span className="text-green-400">+</span> : part.removed ? <span className="text-red-400">-</span> : null}
                </div>
                <div className="flex-1">
                  <SyntaxHighlighter
                    language={language}
                    style={oneDark}
                    PreTag="div"
                    wrapLongLines={false}
                    customStyle={{
                      margin: 0,
                      padding: 0,
                      background: 'transparent',
                    }}
                    codeTagProps={{
                      style: {
                        fontSize: '0.75rem',
                        lineHeight: '1.6',
                      }
                    }}
                  >
                    {value}
                  </SyntaxHighlighter>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

/**
 * Widget for Edit tool result - shows a diff view
 */
export const EditResultWidget: React.FC<{ content: string }> = ({ content }) => {
  // Parse the content to extract file path and code snippet
  const lines = content.split('\n');
  let filePath = '';
  const codeLines: { lineNumber: string; code: string }[] = [];
  
  for (const rawLine of lines) {
    const line = rawLine.replace(/\r$/, '');
    if (line.includes('The file') && line.includes('has been updated')) {
      const match = line.match(/The file (.+) has been updated/);
      if (match) {
        filePath = match[1];
      }
    } else if (/^\s*\d+/.test(line)) {
      const lineMatch = line.match(/^\s*(\d+)\t?(.*)$/);
      if (lineMatch) {
        const [, lineNum, codePart] = lineMatch;
        codeLines.push({
          lineNumber: lineNum,
          code: codePart,
        });
      }
    }
  }

  const codeContent = codeLines.map(l => l.code).join('\n');
  const firstNumberedLine = codeLines.find(l => l.lineNumber !== '');
  const startLineNumber = firstNumberedLine ? parseInt(firstNumberedLine.lineNumber) : 1;

  const getLanguage = (path: string) => {
    const ext = path.split('.').pop()?.toLowerCase();
    const languageMap: Record<string, string> = {
      ts: "typescript", tsx: "tsx", js: "javascript", jsx: "jsx",
      py: "python", rs: "rust", go: "go", java: "java"
    };
    return languageMap[ext || ""] || "text";
  };

  const language = getLanguage(filePath);

  return (
    <div className="rounded-lg border bg-zinc-950 overflow-hidden">
      <div className="px-4 py-2 border-b bg-emerald-950/30 flex items-center gap-2">
        <GitBranch className="h-3.5 w-3.5 text-emerald-500" />
        <span className="text-xs font-mono text-emerald-400">Edit Result</span>
        {filePath && (
          <>
            <ChevronRight className="h-3 w-3 text-muted-foreground" />
            <span className="text-xs font-mono text-muted-foreground">{filePath}</span>
          </>
        )}
      </div>
      <div className="overflow-x-auto max-h-[440px]">
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
    </div>
  );
};

// Export remaining widgets and continue building the comprehensive tool widget system
export const MCPWidget: React.FC<{ 
  toolName: string; 
  input?: any;
  result?: any;
}> = ({ toolName, input, result: _result }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Parse the tool name to extract components
  const parts = toolName.split('__');
  const namespace = parts[1] || '';
  const method = parts[2] || '';
  
  const formatNamespace = (ns: string) => {
    return ns
      .replace(/-/g, ' ')
      .replace(/_/g, ' ')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };
  
  const formatMethod = (m: string) => {
    return m
      .replace(/_/g, ' ')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };
  
  const hasInput = input && Object.keys(input).length > 0;
  const inputString = hasInput ? JSON.stringify(input, null, 2) : '';
  const isLargeInput = inputString.length > 200;

  return (
    <div className="rounded-lg border border-violet-500/20 bg-gradient-to-br from-violet-500/5 to-purple-500/5 overflow-hidden">
      <div className="px-4 py-3 bg-gradient-to-r from-violet-500/10 to-purple-500/10 border-b border-violet-500/20">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="relative">
              <Package2 className="h-4 w-4 text-violet-500" />
              <Sparkles className="h-2.5 w-2.5 text-violet-400 absolute -top-1 -right-1" />
            </div>
            <span className="text-sm font-medium text-violet-600 dark:text-violet-400">MCP Tool</span>
          </div>
          {hasInput && isLargeInput && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="text-violet-500 hover:text-violet-600 transition-colors"
            >
              {isExpanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
            </button>
          )}
        </div>
      </div>
      
      <div className="px-4 py-3 space-y-3">
        <div className="flex items-center gap-2 text-sm">
          <span className="text-violet-500 font-medium">MCP</span>
          <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-purple-600 dark:text-purple-400 font-medium">
            {formatNamespace(namespace)}
          </span>
          <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
          <div className="flex items-center gap-1.5">
            <Zap className="h-3.5 w-3.5 text-violet-500" />
            <code className="text-sm font-mono font-semibold text-foreground">
              {formatMethod(method)}
              <span className="text-muted-foreground">()</span>
            </code>
          </div>
        </div>
        
        {hasInput && (
          <div className={cn(
            "transition-all duration-200",
            !isExpanded && isLargeInput && "max-h-[200px]"
          )}>
            <div className="rounded-lg border bg-zinc-950/50 overflow-hidden">
              <div className="px-3 py-2 border-b bg-zinc-900/50 flex items-center gap-2">
                <Code className="h-3 w-3 text-violet-500" />
                <span className="text-xs font-mono text-muted-foreground">Parameters</span>
              </div>
              <div className={cn(
                "overflow-auto",
                !isExpanded && isLargeInput && "max-h-[150px]"
              )}>
                <SyntaxHighlighter
                  language="json"
                  style={oneDark}
                  customStyle={{
                    margin: 0,
                    padding: '0.75rem',
                    background: 'transparent',
                    fontSize: '0.75rem',
                    lineHeight: '1.5',
                  }}
                  wrapLongLines={false}
                >
                  {inputString}
                </SyntaxHighlighter>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Add remaining widget exports
export const CommandWidget: React.FC<{ 
  commandName: string;
  commandMessage: string;
  commandArgs?: string;
}> = ({ commandName, commandMessage, commandArgs }) => {
  return (
    <div className="rounded-lg border bg-zinc-950/50 overflow-hidden">
      <div className="px-4 py-2 border-b bg-zinc-900/50 flex items-center gap-2">
        <Terminal className="h-3.5 w-3.5 text-blue-500" />
        <span className="text-xs font-mono text-blue-400">Command</span>
      </div>
      <div className="p-3 space-y-1">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">$</span>
          <code className="text-sm font-mono text-foreground">{commandName}</code>
          {commandArgs && (
            <code className="text-sm font-mono text-muted-foreground">{commandArgs}</code>
          )}
        </div>
        {commandMessage && commandMessage !== commandName && (
          <div className="text-xs text-muted-foreground ml-4">{commandMessage}</div>
        )}
      </div>
    </div>
  );
};

export const CommandOutputWidget: React.FC<{ 
  output: string;
  onLinkDetected?: (url: string) => void;
}> = ({ output, onLinkDetected }) => {
  return (
    <div className="rounded-lg border bg-zinc-950/50 overflow-hidden">
      <div className="px-4 py-2 bg-zinc-900/50 flex items-center gap-2">
        <ChevronRight className="h-3 w-3 text-green-500" />
        <span className="text-xs font-mono text-green-400">Output</span>
      </div>
      <div className="p-3">
        <pre className="text-sm font-mono text-zinc-300 whitespace-pre-wrap">
          {output || <span className="text-zinc-500 italic">No output</span>}
        </pre>
      </div>
    </div>
  );
};

export const SummaryWidget: React.FC<{ 
  summary: string;
  leafUuid?: string;
}> = ({ summary, leafUuid }) => {
  return (
    <div className="rounded-lg border border-blue-500/20 bg-blue-500/5 overflow-hidden">
      <div className="px-4 py-3 flex items-start gap-3">
        <div className="mt-0.5">
          <div className="h-8 w-8 rounded-full bg-blue-500/10 flex items-center justify-center">
            <Info className="h-4 w-4 text-blue-500" />
          </div>
        </div>
        <div className="flex-1 space-y-1">
          <div className="text-xs font-medium text-blue-600 dark:text-blue-400">AI Summary</div>
          <p className="text-sm text-foreground">{summary}</p>
          {leafUuid && (
            <div className="text-xs text-muted-foreground mt-2">
              ID: <code className="font-mono">{leafUuid.slice(0, 8)}...</code>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export const MultiEditWidget: React.FC<{
  file_path: string;
  edits: Array<{ old_string: string; new_string: string }>;
  result?: any;
}> = ({ file_path, edits, result: _result }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 mb-2">
        <FileEdit className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm font-medium">Using tool: MultiEdit</span>
      </div>
      <div className="ml-6 space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-blue-500">{file_path}</span>
        </div>
        
        <div className="space-y-1">
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            <ChevronRight className={cn("h-3 w-3 transition-transform", isExpanded && "rotate-90")} />
            {edits.length} edit{edits.length !== 1 ? 's' : ''}
          </button>
          
          {isExpanded && (
            <div className="space-y-3 mt-3">
              {edits.map((edit, index) => (
                <div key={index} className="space-y-1">
                  <div className="text-xs font-medium text-muted-foreground">Edit {index + 1}</div>
                  <div className="text-xs bg-muted p-2 rounded font-mono">
                    Changes: {edit.old_string.length} → {edit.new_string.length} chars
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export const MultiEditResultWidget: React.FC<{ 
  content: string;
  edits?: Array<{ old_string: string; new_string: string }>;
}> = ({ content, edits }) => {
  return (
    <div className="p-3 bg-muted/50 rounded-md border">
      <pre className="text-xs font-mono whitespace-pre-wrap">{content}</pre>
    </div>
  );
};

export const SystemReminderWidget: React.FC<{ message: string }> = ({ message }) => {
  return (
    <div className="flex items-start gap-2 p-3 rounded-md border border-blue-500/20 bg-blue-500/5 text-blue-600">
      <Info className="h-4 w-4 mt-0.5" />
      <div className="flex-1 text-sm">{message}</div>
    </div>
  );
};

export const SystemInitializedWidget: React.FC<{
  sessionId?: string;
  model?: string;
  cwd?: string;
  tools?: string[];
}> = ({ sessionId, model, cwd, tools = [] }) => {
  return (
    <Card className="border-blue-500/20 bg-blue-500/5">
      <CardContent className="p-4">
        <div className="flex items-start gap-3">
          <Info className="h-5 w-5 text-blue-500 mt-0.5" />
          <div className="flex-1 space-y-2">
            <h4 className="font-semibold text-sm">System Initialized</h4>
            {sessionId && <div className="text-xs">Session: {sessionId}</div>}
            {model && <div className="text-xs">Model: {model}</div>}
            {cwd && <div className="text-xs">Working Directory: {cwd}</div>}
            {tools.length > 0 && (
              <div className="text-xs">Tools: {tools.length} available</div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export const TaskWidget: React.FC<{ 
  description?: string; 
  prompt?: string;
  result?: any;
}> = ({ description, prompt, result: _result }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 mb-2">
        <div className="relative">
          <Bot className="h-4 w-4 text-purple-500" />
          <Sparkles className="h-2.5 w-2.5 text-purple-400 absolute -top-1 -right-1" />
        </div>
        <span className="text-sm font-medium">Spawning Sub-Agent Task</span>
      </div>
      
      <div className="ml-6 space-y-3">
        {description && (
          <div className="rounded-lg border border-purple-500/20 bg-purple-500/5 p-3">
            <div className="flex items-center gap-2 mb-1">
              <Zap className="h-3.5 w-3.5 text-purple-500" />
              <span className="text-xs font-medium text-purple-600 dark:text-purple-400">Task Description</span>
            </div>
            <p className="text-sm text-foreground ml-5">{description}</p>
          </div>
        )}
        
        {prompt && (
          <div className="space-y-2">
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              <ChevronRight className={cn("h-3 w-3 transition-transform", isExpanded && "rotate-90")} />
              <span>Task Instructions</span>
            </button>
            
            {isExpanded && (
              <div className="rounded-lg border bg-muted/30 p-3">
                <pre className="text-xs font-mono text-muted-foreground whitespace-pre-wrap">
                  {prompt}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export const ThinkingWidget: React.FC<{ 
  thinking: string;
  signature?: string;
}> = ({ thinking }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const trimmedThinking = thinking.trim();
  
  return (
    <div className="rounded-lg border border-gray-500/20 bg-gray-500/5 overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-gray-500/10 transition-colors"
      >
        <div className="flex items-center gap-2">
          <div className="relative">
            <Bot className="h-4 w-4 text-gray-500" />
            <Sparkles className="h-2.5 w-2.5 text-gray-400 absolute -top-1 -right-1 animate-pulse" />
          </div>
          <span className="text-sm font-medium text-gray-600 dark:text-gray-400 italic">
            Thinking...
          </span>
        </div>
        <ChevronRight className={cn(
          "h-4 w-4 text-gray-500 transition-transform",
          isExpanded && "rotate-90"
        )} />
      </button>
      
      {isExpanded && (
        <div className="px-4 pb-4 pt-2 border-t border-gray-500/20">
          <pre className="text-xs font-mono text-gray-600 dark:text-gray-400 whitespace-pre-wrap bg-gray-500/5 p-3 rounded-lg italic">
            {trimmedThinking}
          </pre>
        </div>
      )}
    </div>
  );
};

export const WebSearchWidget: React.FC<{ 
  query: string; 
  result?: any;
}> = ({ query, result }) => {
  const [expandedSections, setExpandedSections] = useState<Set<number>>(new Set());
  
  const toggleSection = (index: number) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedSections(newExpanded);
  };
  
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-blue-500/5 border border-blue-500/10">
        <Globe className="h-4 w-4 text-blue-500/70" />
        <span className="text-xs font-medium uppercase tracking-wider text-blue-600/70 dark:text-blue-400/70">Web Search</span>
        <span className="text-sm text-muted-foreground/80 flex-1 truncate">{query}</span>
      </div>
      
      {result && (
        <div className="rounded-lg border bg-background/50 backdrop-blur-sm overflow-hidden">
          <div className="p-3">
            <div className="prose prose-sm dark:prose-invert max-w-none">
              <ReactMarkdown>{result.content || 'Search completed'}</ReactMarkdown>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};