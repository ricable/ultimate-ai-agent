/**
 * Session View - Enhanced chat interface for OpenCode sessions
 * Ported from Claudia with multi-provider support and real-time features
 */

"use client";

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ArrowLeft,
  Send,
  Square,
  Paperclip,
  Settings,
  Activity,
  Globe,
  GitBranch,
  Clock,
  Zap,
  AlertCircle,
  CheckCircle,
  Loader2,
  Share2,
  Terminal,
  ChevronRight,
  FileEdit,
  FileText,
  FolderOpen,
  Package,
  Sparkles
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import { useSessionStore, useActiveSession, useActiveSessionMessages } from '@/lib/session-store';
import { Message } from '@/lib/opencode-client';
import { SessionTimeline } from './session-timeline';
import { SessionSharing } from './session-sharing';

interface SessionViewProps {
  sessionId: string;
}

interface ChatMessageProps {
  message: Message;
  isLatest: boolean;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, isLatest }) => {
  const getMessageIcon = () => {
    switch (message.type) {
      case 'user':
        return '👤';
      case 'assistant':
        return '🤖';
      case 'system':
        return '⚙️';
      case 'tool':
        return '🔧';
      default:
        return '💬';
    }
  };

  const formatContent = (content: any) => {
    if (typeof content === 'string') {
      return content;
    }
    
    if (Array.isArray(content)) {
      return content.map((item, index) => {
        if (typeof item === 'string') return item;
        if (item.type === 'text') return item.text;
        if (item.type === 'tool_use') {
          return `[Tool: ${item.name}]`;
        }
        return JSON.stringify(item);
      }).join(' ');
    }
    
    return JSON.stringify(content);
  };

  const formatTimestamp = (timestamp: number) => {
    return new Date(timestamp).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn(
        "flex space-x-3 p-4",
        message.type === 'user' && "bg-muted/50",
        isLatest && "ring-2 ring-primary/20"
      )}
    >
      <div className="flex-shrink-0">
        <div className={cn(
          "w-8 h-8 rounded-full flex items-center justify-center text-sm",
          message.type === 'user' && "bg-primary text-primary-foreground",
          message.type === 'assistant' && "bg-secondary text-secondary-foreground",
          message.type === 'system' && "bg-muted text-muted-foreground",
          message.type === 'tool' && "bg-accent text-accent-foreground"
        )}>
          {getMessageIcon()}
        </div>
      </div>
      
      <div className="flex-1 min-w-0 space-y-2">
        <div className="flex items-center space-x-2">
          <span className={cn(
            "text-sm font-medium capitalize",
            message.type === 'user' && "text-primary",
            message.type === 'assistant' && "text-secondary-foreground",
            message.type === 'system' && "text-muted-foreground",
            message.type === 'tool' && "text-accent-foreground"
          )}>
            {message.type}
          </span>
          <Badge variant="outline" className="text-xs">
            {message.provider}
          </Badge>
          <Badge variant="outline" className="text-xs">
            {message.model}
          </Badge>
          <span className="text-xs text-muted-foreground">
            {formatTimestamp(message.timestamp)}
          </span>
        </div>
        
        <div className="prose prose-sm max-w-none">
          <p className="whitespace-pre-wrap break-words">
            {formatContent(message.content)}
          </p>
        </div>
        
        {message.tokens && (
          <div className="flex items-center space-x-4 text-xs text-muted-foreground">
            <span>Input: {message.tokens.input} tokens</span>
            <span>Output: {message.tokens.output} tokens</span>
          </div>
        )}
        
        {message.tool_calls && message.tool_calls.length > 0 && (
          <div className="space-y-2">
            {/* Tool widgets from Claudia */}
            {message.tool_calls.map((tool, index) => {
              // Render specific tool widgets based on tool name
              const toolName = tool.name.toLowerCase();
              
              if (toolName === 'bash') {
                return (
                  <div key={index} className="rounded-lg border bg-zinc-950 overflow-hidden">
                    <div className="px-4 py-2 bg-zinc-900/50 flex items-center gap-2 border-b">
                      <Terminal className="h-3.5 w-3.5 text-green-500" />
                      <span className="text-xs font-mono text-muted-foreground">Terminal</span>
                      {tool.input?.description && (
                        <>
                          <ChevronRight className="h-3 w-3 text-muted-foreground" />
                          <span className="text-xs text-muted-foreground">{tool.input.description}</span>
                        </>
                      )}
                    </div>
                    <div className="p-4">
                      <code className="text-xs font-mono text-green-400 block">
                        $ {tool.input?.command || 'command'}
                      </code>
                    </div>
                  </div>
                );
              }
              
              if (toolName.includes('edit') || toolName === 'write') {
                return (
                  <div key={index} className="space-y-2">
                    <div className="flex items-center gap-2 p-3 rounded-lg bg-muted/50">
                      <FileEdit className="h-4 w-4 text-primary" />
                      <span className="text-sm">
                        {toolName === 'write' ? 'Writing to file:' : 'Editing file:'}
                      </span>
                      <code className="text-sm font-mono bg-background px-2 py-0.5 rounded">
                        {tool.input?.file_path || tool.input?.filePath || 'file'}
                      </code>
                    </div>
                  </div>
                );
              }
              
              if (toolName === 'read') {
                return (
                  <div key={index} className="flex items-center gap-2 p-3 rounded-lg bg-muted/50">
                    <FileText className="h-4 w-4 text-primary" />
                    <span className="text-sm">Reading file:</span>
                    <code className="text-sm font-mono bg-background px-2 py-0.5 rounded">
                      {tool.input?.file_path || tool.input?.filePath || 'file'}
                    </code>
                  </div>
                );
              }
              
              if (toolName === 'ls') {
                return (
                  <div key={index} className="flex items-center gap-2 p-3 rounded-lg bg-muted/50">
                    <FolderOpen className="h-4 w-4 text-primary" />
                    <span className="text-sm">Listing directory:</span>
                    <code className="text-sm font-mono bg-background px-2 py-0.5 rounded">
                      {tool.input?.path || 'directory'}
                    </code>
                  </div>
                );
              }
              
              if (toolName.includes('mcp__')) {
                const parts = tool.name.split('__');
                const namespace = parts[1] || '';
                const method = parts[2] || '';
                
                return (
                  <div key={index} className="rounded-lg border border-violet-500/20 bg-gradient-to-br from-violet-500/5 to-purple-500/5 overflow-hidden">
                    <div className="px-4 py-3 bg-gradient-to-r from-violet-500/10 to-purple-500/10 border-b border-violet-500/20">
                      <div className="flex items-center gap-2">
                        <div className="relative">
                          <Package className="h-4 w-4 text-violet-500" />
                          <Sparkles className="h-2.5 w-2.5 text-violet-400 absolute -top-1 -right-1" />
                        </div>
                        <span className="text-sm font-medium text-violet-600 dark:text-violet-400">MCP Tool</span>
                      </div>
                    </div>
                    <div className="px-4 py-3">
                      <div className="flex items-center gap-2 text-sm">
                        <span className="text-violet-500 font-medium">MCP</span>
                        <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                        <span className="text-purple-600 dark:text-purple-400 font-medium">
                          {namespace.charAt(0).toUpperCase() + namespace.slice(1)}
                        </span>
                        <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                        <code className="text-sm font-mono font-semibold text-foreground">
                          {method}()
                        </code>
                      </div>
                    </div>
                  </div>
                );
              }
              
              // Default tool display
              return (
                <Card key={index} className="bg-muted/50">
                  <CardContent className="p-3">
                    <div className="flex items-center space-x-2 mb-2">
                      <Zap className="h-4 w-4" />
                      <span className="text-sm font-medium">{tool.name}</span>
                    </div>
                    <pre className="text-xs bg-background p-2 rounded overflow-x-auto">
                      {JSON.stringify(tool.input, null, 2)}
                    </pre>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}
      </div>
    </motion.div>
  );
};

export const SessionView: React.FC<SessionViewProps> = ({ sessionId }) => {
  const session = useActiveSession();
  const messages = useActiveSessionMessages();
  const {
    isStreaming,
    streamingSessionId,
    providers,
    actions
  } = useSessionStore();

  const [inputValue, setInputValue] = useState('');
  const [isComposing, setIsComposing] = useState(false);
  const [showTimeline, setShowTimeline] = useState(false);
  const [showSharing, setShowSharing] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const isCurrentSessionStreaming = isStreaming && streamingSessionId === sessionId;

  // Load session messages on mount
  useEffect(() => {
    if (sessionId) {
      actions.loadSessionMessages(sessionId);
    }
  }, [sessionId]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollElement = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollElement) {
        scrollElement.scrollTop = scrollElement.scrollHeight;
      }
    }
  }, [messages.length]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 200) + 'px';
    }
  }, [inputValue]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isCurrentSessionStreaming) return;

    const messageContent = inputValue.trim();
    setInputValue('');

    try {
      await actions.sendMessage(sessionId, messageContent);
    } catch (error) {
      console.error('Failed to send message:', error);
      // Restore input value on error
      setInputValue(messageContent);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !isComposing) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const getCurrentProvider = () => {
    if (!session) return null;
    return providers.find(p => p.id === session.provider);
  };

  const currentProvider = getCurrentProvider();

  // Enhanced message formatting with better tool detection
  const shouldShowToolWidget = (message: Message) => {
    if (!message.tool_calls || message.tool_calls.length === 0) return false;
    
    // Show widgets for these tools
    const toolsWithWidgets = [
      'bash', 'edit', 'multiedit', 'write', 'read', 'ls', 'glob', 'grep',
      'todowrite', 'websearch', 'task', 'mcp__'
    ];
    
    return message.tool_calls.some(tool => 
      toolsWithWidgets.some(widgetTool => tool.name.toLowerCase().includes(widgetTool))
    );
  };

  if (!session) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center space-y-4">
          <AlertCircle className="h-12 w-12 text-muted-foreground mx-auto" />
          <h3 className="text-lg font-medium">Session Not Found</h3>
          <p className="text-muted-foreground">
            The requested session could not be loaded.
          </p>
          <Button onClick={() => actions.setCurrentView('projects')}>
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Projects
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => actions.setCurrentView('projects')}
              className="h-8 w-8"
            >
              <ArrowLeft className="h-4 w-4" />
            </Button>
            
            <div>
              <h1 className="text-xl font-semibold">
                {session.name || `Session ${session.id.slice(0, 8)}`}
              </h1>
              <div className="flex items-center space-x-2 text-sm text-muted-foreground">
                <span>{session.project_path}</span>
                <span>•</span>
                <div className="flex items-center space-x-1">
                  <div className={cn(
                    "h-2 w-2 rounded-full",
                    currentProvider?.status === 'online' ? 'bg-green-500' : 'bg-red-500'
                  )} />
                  <span>{session.provider}</span>
                </div>
                <span>•</span>
                <span>{session.model}</span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button 
                    variant="outline" 
                    size="icon"
                    onClick={() => setShowTimeline(true)}
                  >
                    <GitBranch className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Timeline & Checkpoints</TooltipContent>
              </Tooltip>
            </TooltipProvider>
            
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button 
                    variant="outline" 
                    size="icon"
                    onClick={() => setShowSharing(true)}
                  >
                    <Share2 className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Share Session</TooltipContent>
              </Tooltip>
            </TooltipProvider>
            
            <Button variant="outline" size="icon">
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </div>
        
        {/* Session Stats */}
        <div className="flex items-center space-x-6 mt-3 text-xs text-muted-foreground">
          <div className="flex items-center space-x-1">
            <Clock className="h-3 w-3" />
            <span>
              Updated {new Date(session.updated_at).toLocaleDateString()}
            </span>
          </div>
          <div className="flex items-center space-x-1">
            <Activity className="h-3 w-3" />
            <span>{messages.length} messages</span>
          </div>
          {session.token_usage && (
            <div className="flex items-center space-x-1">
              <Zap className="h-3 w-3" />
              <span>
                {session.token_usage.input_tokens + session.token_usage.output_tokens} tokens
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-hidden">
        <ScrollArea ref={scrollAreaRef} className="h-full">
          <div className="max-w-4xl mx-auto">
            {messages.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center space-y-4">
                  <div className="text-4xl">👋</div>
                  <h3 className="text-lg font-medium">Start the conversation</h3>
                  <p className="text-muted-foreground max-w-md">
                    Ask anything about your code, request changes, or describe what you&apos;d like to build.
                  </p>
                </div>
              </div>
            ) : (
              <div className="divide-y divide-border">
                <AnimatePresence>
                  {messages.map((message, index) => (
                    <ChatMessage
                      key={message.id}
                      message={message}
                      isLatest={index === messages.length - 1}
                    />
                  ))}
                </AnimatePresence>
                
                {isCurrentSessionStreaming && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex items-center space-x-3 p-4"
                  >
                    <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center">
                      🤖
                    </div>
                    <div className="flex items-center space-x-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span className="text-sm text-muted-foreground">
                        AI is thinking...
                      </span>
                    </div>
                  </motion.div>
                )}
              </div>
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Input Area */}
      <div className="border-t border-border p-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-end space-x-3">
            <div className="flex-1 relative">
              <Textarea
                ref={textareaRef}
                placeholder="Ask me anything about your code..."
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyPress}
                onCompositionStart={() => setIsComposing(true)}
                onCompositionEnd={() => setIsComposing(false)}
                disabled={isCurrentSessionStreaming}
                className="min-h-[44px] max-h-[200px] resize-none pr-12"
                rows={1}
              />
              <Button
                variant="ghost"
                size="icon"
                className="absolute bottom-2 right-2 h-8 w-8"
                disabled={isCurrentSessionStreaming}
              >
                <Paperclip className="h-4 w-4" />
              </Button>
            </div>
            
            <Button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isCurrentSessionStreaming}
              size="icon"
              className="h-11 w-11"
            >
              {isCurrentSessionStreaming ? (
                <Square className="h-4 w-4" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
          
          <div className="flex items-center justify-between mt-2 text-xs text-muted-foreground">
            <div className="flex items-center space-x-4">
              <span>Press Enter to send, Shift+Enter for new line</span>
              {currentProvider && (
                <div className="flex items-center space-x-1">
                  <CheckCircle className="h-3 w-3 text-green-500" />
                  <span>{currentProvider.name} connected</span>
                </div>
              )}
            </div>
            {inputValue.length > 0 && (
              <span>{inputValue.length} characters</span>
            )}
          </div>
        </div>
      </div>

      {/* Timeline Dialog */}
      {showTimeline && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center">
          <div className="bg-background border rounded-lg shadow-lg w-full max-w-2xl h-[80vh] relative">
            <div className="absolute top-4 right-4 z-10">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setShowTimeline(false)}
              >
                <ArrowLeft className="h-4 w-4" />
              </Button>
            </div>
            <SessionTimeline sessionId={sessionId} />
          </div>
        </div>
      )}

      {/* Sharing Dialog */}
      {session && (
        <SessionSharing
          session={session}
          open={showSharing}
          onOpenChange={setShowSharing}
        />
      )}
    </div>
  );
};